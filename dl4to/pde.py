"""Cleaned and reconstructed PDE solver module with multi-backend SparseLinearSolver (scipy, cupy, legate, amgx, mfem)."""

__all__ = [
    'AutogradLinearSolver', 'LinearSolver', 'SparseLinearSolver', 'PDESolver',
    'FDMDerivatives', 'FDMAdjointDerivatives', 'FDMAssembly', 'UnpaddedFDM', 'FDM',
    'configure_legion_memory'
]
from typing import Callable, Any
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import torch, numpy as np, os, copy, ctypes, sys  # added required imports
import multiprocessing as mp
from multiprocessing.managers import SyncManager
import socket
import uuid
import time
from queue import Empty

# Optional UMFPACK factorization
try:
    from scikits.umfpack import factorized
except Exception:
    from scipy.sparse.linalg import factorized

# Optional backend availability flags
try:
    import cupy  # noqa: F401
    HAVE_CUPY = torch.cuda.is_available()
except Exception:
    HAVE_CUPY = False


try:
    import legate_sparse  # noqa: F401
    HAVE_LEGATE_SPARSE = True
except Exception:
    HAVE_LEGATE_SPARSE = False

try:
    import pyamgx  # noqa: F401
    HAVE_PYAMGX = True
except Exception:
    HAVE_PYAMGX = False

try:  # PETSc optional GPU backend
    _petsc_dir = os.getenv('PETSC_DIR')
    if _petsc_dir:
        _py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        _petsc_paths = [
            os.path.join(_petsc_dir, 'lib', _py_ver, 'site-packages'),
            os.path.join(_petsc_dir, 'lib')
        ]
        for _pth in _petsc_paths:
            if os.path.isdir(_pth) and _pth not in sys.path:
                sys.path.append(_pth)
    import petsc4py  # type: ignore
    from petsc4py import PETSc  # type: ignore
    HAVE_PETSC4PY = True
except Exception:
    HAVE_PETSC4PY = False
    PETSc = None  # type: ignore
    petsc4py = None  # type: ignore

# Optional torch-fem backend
try:
    from torchfem.sparse import sparse_solve as tfem_sparse_solve
    from torchfem.sparse import CachedSolve as TFEMCachedSolve
    HAVE_TORCHFEM = True
except Exception:
    HAVE_TORCHFEM = False

# Global singleton for MFEM device to avoid reconfiguration aborts
_MFEM_DEVICE_SINGLETON = None

# Lightweight cache for MFEM SyncManager connection/proxies to avoid reconnect per call
_MFEM_Q_MANAGER = None
_MFEM_Q_INPUT = None
_MFEM_Q_OUTPUT = None
_MFEM_Q_ADDR = None  # tuple (host, port)


def configure_legion_memory(replheap_size_mb: int = 64):
    """Set Legion replheap size (helpful for legate-sparse)."""
    os.environ.setdefault("LEGATE_REPL_HEAP_SIZE", str(replheap_size_mb))


class AutogradLinearSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, θ, A_op, b, solver, A_mat, factorize=True):
        """
        In the forward pass we receive a tensor containing the input and return
        a tensor containing the output. `ctx` is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the `ctx.save_for_backward` method.

        Returns
        torch.Tensor
        """
        np_b = b.cpu().numpy()
        # Explicit phase label
        phase = 'forward'

        if factorize:
            solver = factorized(A_mat)
            x = solver(np_b)
        else:
            # pass phase explicitly and include θ for warm-start gating if solver supports it
            try:
                x = solver(A_mat, np_b, phase=phase, θ_current=θ.detach().cpu().numpy())
            except TypeError:
                # Backward compatibility if older signature
                x = solver(A_mat, np_b, phase=phase)

        x = torch.from_numpy(x.astype(np_b.dtype))
        ctx.save_for_backward(θ, x, b)
        ctx.intermediate = (A_mat, solver, A_op, factorize)
        return x


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        Returns
        ----------
        (torch.Tensor, None, None, None, None, None)
        """
        torch.set_grad_enabled(True)
        θ, x, b = ctx.saved_tensors
        A_mat, solver, A_op, factorize = ctx.intermediate
        θ = θ.clone().detach()
        θ.requires_grad_(True)

        with torch.no_grad():
            phase = 'adjoint'
            flat_np_grad_output = grad_output.flatten().cpu().numpy()
            t_adj0 = _pde_now()
            if factorize:
                y = solver(flat_np_grad_output)
            else:
                try:
                    y = solver(A_mat, flat_np_grad_output, phase=phase, θ_current=θ.detach().cpu().numpy())
                except TypeError:
                    y = solver(A_mat, flat_np_grad_output, phase=phase)
            _pde_log("adjoint_solve", t_adj0)
            y = torch.from_numpy(y).clone().requires_grad_(False)
            x = x.clone().requires_grad_(False)
        t_Aop0 = _pde_now()
        expr = torch.sum(y * (b - A_op(x, θ).flatten()))
        _pde_log("Aop_residual", t_Aop0)
        t_grad0 = _pde_now()
        grad_input = torch.autograd.grad(expr, θ)
        _pde_log("grad_theta", t_grad0)
        return grad_input[0], None, None, None, None, None


class LinearSolver:
    def __init__(self, factorize: bool = True):
        self.autograd_linear_solver = AutogradLinearSolver.apply
        self.factorize = factorize

    def _backend(self):
        raise NotImplementedError

    def __call__(self, θ: torch.Tensor, A_op: Callable, b: torch.Tensor, A_mat):
        return self.autograd_linear_solver(θ, A_op, b, self._backend(), A_mat, self.factorize)


class SparseLinearSolver(LinearSolver):
    def __init__(self,
                 optimizer: str = 'scipy',
                 use_umfpack: bool = True,
                 factorize: bool = False,
                 cg_tol: float = 1e-6,
                 cg_max_iter: int | None = None,
                 preconditioner: str = 'jacobi',  # 'none' | 'jacobi' | 'legat'
                 amgx_config: dict | None = None):
        self.optimizer = optimizer.lower()
        self.use_umfpack = use_umfpack
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.preconditioner = preconditioner.lower()
        self.amgx_config = amgx_config
        # Optional initial guess for iterative solvers (used by 'mfem' backend)
        self.initial_guess = None
        # torch-fem cached solve for warm starts
        self._tfem_cache = TFEMCachedSolve() if 'HAVE_TORCHFEM' in globals() and HAVE_TORCHFEM else None
        # Optional RBM modes for torch-fem AMG preconditioner
        self._tfem_B = None  # type: ignore[assignment]
        super().__init__(factorize=factorize)

    # Public name retained for compatibility with earlier code
    def _solver(self):  # legacy name
        return self._backend()

    def _backend(self):
        opt = self.optimizer

        # --- MFEM distributed backend (with optional PyMETIS partition) ---
        if opt == 'mfem':
            # Automatic values-only mode: after first full matrix with a given sparsity pattern
            # subsequent calls send only updated numeric values (and RHS). Can be disabled by MFEM_NO_VALUES_ONLY=1.
            auto_disable = os.getenv('MFEM_NO_VALUES_ONLY', '0') == '1'
            reuse_prec = os.getenv('MFEM_REUSE_PREC', '0') == '1'
            # Pattern cache for values-only mode (structure reuse) stored on solver instance
            if not hasattr(self, '_mfem_pattern_sig'):
                self._mfem_pattern_sig = None  # (shape, nnz, hash)
                self._mfem_pattern_indices = None
                self._mfem_pattern_indptr = None
            def solve_mfem_remote(A, b, phase='forward', θ_current=None):
                """
                Remote MFEM solver using client-server architecture.
                Connects to mfem_solver_process.py via SyncManager queues.
                """
                try:
                    # Minimal, opt-in client-side comm profiling
                    _prof = os.getenv("MFEM_COMM_PROFILE", "0") == "1"
                    _log_path = os.getenv("MFEM_CLIENT_TIMINGS", "saved_solutions/mfem_client_timings.jsonl")
                    import time as _time
                    import json as _json
                    from pathlib import Path as _Path
                    _use_shm = os.getenv("MFEM_USE_SHM", "0") == "1"
                    _shm_handles = []  # to hold client-owned shared memory until response
                    def _shm_create_from(arr, name_prefix):
                        from multiprocessing import shared_memory as _shm
                        a = np.ascontiguousarray(arr)
                        shm = _shm.SharedMemory(create=True, size=a.nbytes)
                        _shm_handles.append(shm)
                        # write bytes
                        mv = memoryview(shm.buf)
                        mv[:a.nbytes] = a.view(np.uint8)
                        return shm
                    def _t_ms(t0):
                        return (_time.perf_counter() - t0) * 1000.0
                    def _log_row(row: dict):
                        try:
                            p = _Path(_log_path)
                            p.parent.mkdir(parents=True, exist_ok=True)
                            with open(p, 'a') as f:
                                f.write(_json.dumps(row) + "\n")
                        except Exception:
                            pass

                    # Connect to MFEM solver server
                    port = 50000
                    authkey = b'mfem_solver'
                    
                    class QueueManager(SyncManager):
                        pass
                    
                    QueueManager.register('get_input_queue')
                    QueueManager.register('get_output_queue')

                    # Reuse existing connection if available and same address
                    global _MFEM_Q_MANAGER, _MFEM_Q_INPUT, _MFEM_Q_OUTPUT, _MFEM_Q_ADDR
                    if _MFEM_Q_MANAGER is None or _MFEM_Q_ADDR != ('localhost', port):
                        try:
                            _t0 = time.time()
                            manager = QueueManager(address=('localhost', port), authkey=authkey)
                            manager.connect()
                            input_queue = manager.get_input_queue()
                            output_queue = manager.get_output_queue()
                            # cache
                            _MFEM_Q_MANAGER = manager
                            _MFEM_Q_INPUT = input_queue
                            _MFEM_Q_OUTPUT = output_queue
                            _MFEM_Q_ADDR = ('localhost', port)
                            if _prof:
                                _connect_ms = (time.time() - _t0) * 1000.0
                                print(f"[mfem-comm] connect_ms={_connect_ms:.2f}", flush=True)
                        except (ConnectionRefusedError, socket.error) as e:
                            raise ValueError(f"[mfem] Failed to connect to server on port {port}: {e}")
                    else:
                        input_queue = _MFEM_Q_INPUT
                        output_queue = _MFEM_Q_OUTPUT
                        if _prof:
                            print("[mfem-comm] reused_conn=1", flush=True)
                    
                        
                    
                    # Convert matrix to CSR format for serialization
                    t_asm0 = _time.perf_counter()
                    A_csr = A.tocsr() if hasattr(A, 'tocsr') else csr_matrix(A)
                    if A_csr.dtype != np.float64:
                        A_csr = A_csr.astype(np.float64)
                    # Reduce payload size: ensure index arrays are int32 (common SciPy default, half the bytes vs int64)
                    indices_i32 = np.asarray(A_csr.indices, dtype=np.int32)
                    indptr_i32 = np.asarray(A_csr.indptr, dtype=np.int32)
                    use_values_only = False
                    pattern_sig = None
                    if not auto_disable:
                        try:
                            import hashlib as _hashlib
                            h = _hashlib.blake2b(digest_size=8)
                            h.update(indices_i32.tobytes())
                            h.update(indptr_i32.tobytes())
                            pattern_sig = (A_csr.shape, int(A_csr.nnz), h.hexdigest())
                            if self._mfem_pattern_sig == pattern_sig:
                                use_values_only = True
                            else:
                                self._mfem_pattern_sig = pattern_sig
                                self._mfem_pattern_indices = indices_i32.copy()
                                self._mfem_pattern_indptr = indptr_i32.copy()
                        except Exception:
                            use_values_only = False
                    t_asm_ms = _t_ms(t_asm0)
                    
                    b_np = np.asarray(b, dtype=np.float64).reshape(-1)
                    
                    # Optional initial guess (x0)
                    x0_np = None
                    try:
                        if isinstance(self.initial_guess, np.ndarray) and self.initial_guess.size == A_csr.shape[0]:
                            x0_np = np.asarray(self.initial_guess, dtype=np.float64).reshape(-1)
                    except Exception:
                        x0_np = None

                    # Estimate bytes to send (dominant payload only)
                    if use_values_only:
                        _bytes_out = int(A_csr.data.nbytes + b_np.nbytes)
                    else:
                        _bytes_out = int(A_csr.data.nbytes + indices_i32.nbytes + indptr_i32.nbytes + b_np.nbytes)
                    if x0_np is not None:
                        _bytes_out += int(x0_np.nbytes)

                    # Generate unique request ID
                    request_id = str(uuid.uuid4())
                    
                    # Send request to server (include phase for preconditioner caching)
                    if use_values_only:
                        # Only send numeric values (and rhs / initial guess) – structure assumed cached on server
                        if _use_shm:
                            sh_data = _shm_create_from(A_csr.data, "A_data_vals")
                            sh_b = _shm_create_from(b_np, "b")
                            sh_x0 = _shm_create_from(x0_np, "x0") if x0_np is not None else None
                            payload = {
                                'transport': 'shm_v1',
                                'values_only': True,
                                'A_shape': tuple(map(int, A_csr.shape)),
                                'nnz': int(A_csr.nnz),
                                'dtypes': {
                                    'data': A_csr.data.dtype.str,
                                    'b': b_np.dtype.str,
                                    'x0': x0_np.dtype.str if x0_np is not None else None,
                                },
                                'names': {
                                    'data': sh_data.name,
                                    'b': sh_b.name,
                                    'x0': (sh_x0.name if sh_x0 is not None else None),
                                },
                                'request_id': request_id,
                                'phase': phase,
                            }
                            _t_put0 = _time.perf_counter(); input_queue.put(payload)
                        else:
                            if x0_np is not None:
                                data = ('VALUES_ONLY', A_csr.data, b_np, request_id, phase, x0_np)
                            else:
                                data = ('VALUES_ONLY', A_csr.data, b_np, request_id, phase)
                            _t_put0 = _time.perf_counter(); input_queue.put(data)
                    else:
                        if _use_shm:
                            # Build shared memory payload (single-rank optimized)
                            sh_data = _shm_create_from(A_csr.data, "A_data")
                            sh_indices = _shm_create_from(indices_i32, "A_indices")
                            sh_indptr = _shm_create_from(indptr_i32, "A_indptr")
                            sh_b = _shm_create_from(b_np, "b")
                            sh_x0 = _shm_create_from(x0_np, "x0") if x0_np is not None else None
                            payload = {
                                'transport': 'shm_v1',
                                'A_shape': tuple(map(int, A_csr.shape)),
                                'nnz': int(A_csr.nnz),
                                'dtypes': {
                                    'data': A_csr.data.dtype.str,
                                    'indices': np.dtype(np.int32).str,
                                    'indptr': np.dtype(np.int32).str,
                                    'b': b_np.dtype.str,
                                    'x0': x0_np.dtype.str if x0_np is not None else None,
                                },
                                'names': {
                                    'data': sh_data.name,
                                    'indices': sh_indices.name,
                                    'indptr': sh_indptr.name,
                                    'b': sh_b.name,
                                    'x0': (sh_x0.name if sh_x0 is not None else None),
                                },
                                'request_id': request_id,
                                'phase': phase,
                            }
                            _t_put0 = _time.perf_counter()
                            input_queue.put(payload)
                        else:
                            if x0_np is not None:
                                data = (A_csr.data, indices_i32, indptr_i32, A_csr.shape, b_np, request_id, phase, x0_np)
                            else:
                                data = (A_csr.data, indices_i32, indptr_i32, A_csr.shape, b_np, request_id, phase)
                            _t_put0 = _time.perf_counter()
                            input_queue.put(data)
                    _put_ms = _t_ms(_t_put0)
                    
                    # Wait for response with timeout
                    _t_wait0 = _time.perf_counter()
                    timeout = 900  # 15 minutes
                    
                    while True:
                        try:
                            result = output_queue.get(timeout=1.0)
                            if len(result) == 3:  # Error case
                                x, req_id, error = result
                                if req_id == request_id:
                                    raise ValueError(f"[mfem] Server error: {error}")
                            else:  # Success case
                                x, req_id = result
                                if req_id == request_id:
                                    _wait_ms = _t_ms(_t_wait0)
                                    x_arr = np.asarray(x)
                                    _bytes_in = int(x_arr.nbytes)
                                    if _prof:
                                        print(
                                            f"[mfem-comm] asm_ms={t_asm_ms:.2f} put_ms={_put_ms:.2f} "
                                            f"wait_ms={_wait_ms:.2f} bytes_out={_bytes_out} bytes_in={_bytes_in}",
                                            flush=True,
                                        )
                                        _log_row({
                                            "phase": phase,
                                            "asm_ms": round(t_asm_ms, 3),
                                            "put_ms": round(_put_ms, 3),
                                            "wait_ms": round(_wait_ms, 3),
                                            "bytes_out": int(_bytes_out),
                                            "bytes_in": int(_bytes_in),
                                        })
                                    # Clear initial guess after use to avoid unintended reuse
                                    self.initial_guess = None
                                    # Cleanup shared memory segments we created
                                    if _use_shm:
                                        try:
                                            for shm in _shm_handles:
                                                try:
                                                    shm.close()
                                                    shm.unlink()
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                                    return x
                        except Empty:
                            if (_time.perf_counter() - _t_wait0) > timeout:
                                raise ValueError("[mfem] Timeout waiting for server response")
                            continue
                            
                except Exception as e:
                    raise ValueError(f"[mfem] Remote solver failed: {e}")
                    
            return solve_mfem_remote

        # --- AMGX ---
        if opt == 'amgx' and HAVE_PYAMGX:
            import pyamgx
            def solve_amgx(A, b, phase='forward', θ_current=None):
                cfg = self.amgx_config or {
                    'config_version': 2,
                    'determinism_flag': 1,
                    'solver': {
                        'preconditioner': {'algorithm': 'AGGREGATION', 'solver': 'AMG'},
                        'solver': 'PCG', 'max_iters': self.cg_max_iter or 100,
                        'tolerance': self.cg_tol, 'norm': 'L2'
                    }
                }
                cfg_obj = pyamgx.Config().create_from_dict(cfg)
                rsrc = pyamgx.Resources().create_simple(cfg_obj)
                solver = pyamgx.Solver().create(rsrc, cfg_obj)
                mode = 'dDDI'
                Am = pyamgx.Matrix().create(rsrc, mode); Am.upload_CSR(A.tocsr())
                n = A.shape[0]
                xb = pyamgx.Vector().create(rsrc, mode); bb = pyamgx.Vector().create(rsrc, mode)
                xb.upload(np.zeros(n, dtype=np.float64)); bb.upload(b.astype(np.float64))
                solver.solve(bb, xb); x = xb.download()
                xb.destroy(); bb.destroy(); Am.destroy(); solver.destroy(); rsrc.destroy(); cfg_obj.destroy()
                return x
            return solve_amgx

        # --- CuPy ---
        if opt == 'cupy' and HAVE_CUPY:
            import cupy as cp
            from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
            from cupyx.scipy.sparse.linalg import cg as cp_cg, LinearOperator
            def solve_cupy(A, b, phase='forward', θ_current=None):
                A_gpu = cp_csc_matrix((cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)), shape=A.shape)
                M = None
                if self.preconditioner == 'jacobi':
                    diag = A_gpu.diagonal(); inv_diag = cp.where(diag != 0, 1.0/diag, 1.0)
                    M = LinearOperator(A_gpu.shape, matvec=lambda x: inv_diag * x, dtype=A_gpu.dtype)
                b_gpu = cp.asarray(b, dtype=cp.float64)
                x_gpu, info = cp_cg(A_gpu, b_gpu, tol=self.cg_tol, maxiter=self.cg_max_iter, M=M)
                cp.cuda.get_current_stream().synchronize()
                return cp.asnumpy(x_gpu)
            return solve_cupy

        # --- Legate Sparse ---
        if opt == 'legate' and HAVE_LEGATE_SPARSE:
            import cupynumeric as cn
            from legate_sparse import csr_matrix as ls_csr
            from legate_sparse.linalg import gmres as ls_gmres, LinearOperator as ls_LinearOperator
            def solve_legate(A, b, phase='forward', θ_current=None):
                import numpy as _np
                data_cn = cn.asarray(_np.asarray(A.data, dtype=_np.float64))
                indices_cn = cn.asarray(_np.asarray(A.indices, dtype=_np.uint64))
                indptr_cn = cn.asarray(_np.asarray(A.indptr, dtype=_np.uint64))
                A_ls = ls_csr((data_cn, indices_cn, indptr_cn), shape=A.shape)
                b_cn = cn.asarray(_np.asarray(b, dtype=_np.float64))
                M = None
                if self.preconditioner == 'jacobi':
                    diag = A_ls.diagonal(); inv_diag = cn.where(diag != 0, 1.0/diag, 1.0)
                    M = ls_LinearOperator(A.shape, matvec=lambda x: inv_diag * x, dtype=_np.float64)
                x_cn, info = ls_gmres(A_ls, b_cn, tol=self.cg_tol, maxiter=self.cg_max_iter, M=M)
                return _np.asarray(x_cn)
            return solve_legate

        # --- torch-fem sparse backend ---
        if opt in ('torch-fem', 'torch_fem', 'torchfem'):
            if not HAVE_TORCHFEM:
                def _err(*args, **kwargs):
                    raise ImportError("torch-fem is not installed. Install with 'pip install torch-fem' in your environment.")
                return _err

            def solve_torchfem(A, b, phase='forward', θ_current=None):
                # Convert SciPy sparse to torch sparse COO
                A_csr = A.tocsr() if hasattr(A, 'tocsr') else csr_matrix(A)
                A_coo = A_csr.tocoo()
                indices = torch.tensor(
                    np.vstack([A_coo.row, A_coo.col]), dtype=torch.long
                )
                values = torch.tensor(A_coo.data, dtype=torch.float64)
                A_t = torch.sparse_coo_tensor(indices, values, size=A_coo.shape)

                b_t = torch.tensor(b, dtype=torch.float64)

                # Choose device automatically; allow override via env
                device_env = os.getenv('TORCHFEM_DEVICE')
                device = device_env if device_env in (None, 'cpu', 'cuda') else None
                if device is None:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # Method selection: prefer cg for large systems, else spsolve
                method = None
                try:
                    n = A_t.shape[0]
                    method = 'cg' if n >= 10000 else 'spsolve'
                except Exception:
                    method = None

                # Warm start via CachedSolve
                cache = self._tfem_cache or TFEMCachedSolve()
                if isinstance(self.initial_guess, np.ndarray) and self.initial_guess.size == A_csr.shape[0]:
                    try:
                        cache.update_x(torch.tensor(self.initial_guess, dtype=torch.float64))
                    except Exception:
                        pass

                # Solve
                x_t = tfem_sparse_solve(
                    A=A_t.coalesce(),
                    b=b_t,
                    B=self._tfem_B if (self._tfem_B is not None) else None,
                    rtol=float(self.cg_tol),
                    device=device,
                    method=('minres' if self._tfem_B is not None else method),
                    M=None,
                    cached_solve=cache,
                    update_cache=True if phase == 'forward' else False,
                )

                # Clear one-shot initial guess
                self.initial_guess = None
                # Persist cache on solver instance
                self._tfem_cache = cache

                return x_t.detach().cpu().numpy()

            return solve_torchfem

    # --- SciPy CPU direct (spsolve) ---
        def solve_cpu(A, b, phase='forward', θ_current=None):
            return spsolve(A, b, use_umfpack=self.use_umfpack)
        return solve_cpu



# Internal Cell
import copy
import torch

# Cell
class PDESolver:
    """
    A parent class that inherits all PDE solvers.
    """
    def __init__(self,
                assemble_tensors_when_passed_to_problem:bool=False # Whether the PDE solver methods pre-assembles any tensors or arrays before solving the PDE for a concrete problem.
                ):
        self.assemble_tensors_when_passed_to_problem = assemble_tensors_when_passed_to_problem


    def __call__(self,
                solution, # The solution for which the PDE should be solved.
                p:float=1., # The SIMP exponent when solving the PDE. Should usually be left at its default value of `1.`.
                binary:bool=False # Whether the densities in the solution should be binarized before solving the PDE.
                ):
        """
        Does the same as the `solve_pde` method. Solves the pde for `solution` and SIMP exponent `p`. Returns three `torch.Tensor` objects: displacements `u`, stresses `σ` and von Mises stresses `σ_vm`.
        """
        return self.solve_pde(solution, p=p, binary=binary)


    def solve_pde(self,
                solution, # The solution for which the PDE should be solved.
                p:float=1., # The SIMP exponent when solving the PDE. Should usually be left at its default value of `1.`.
                binary:bool=False # Whether the densities in the solution should be binarized before solving the PDE.
                ):
        """
        Solves the pde for `solution` and SIMP exponent `p`. Returns three `torch.Tensor` objects: displacements `u`, stresses `σ` and von Mises stresses `σ_vm`.
        """
        raise NotImplementedError("Must be overridden.")


    def clone(self):
        """
        Returns a `dl4to.pde.PDESolver` object, which is a deepcopy of the PDE solver.
        """
        return copy.deepcopy(self)

# Internal Cell
import torch
import torch.autograd.functional as F

# Cell
class FDMDerivatives():
    @staticmethod
    def du_dx_central(u, h):
        du = torch.zeros_like(u)
        du[:, 1:-1, :,:] = (u[:,  2:, :,:] - u[:, 0:-2, :,:]) / (2 * h[0])
        du[:,  0  , :,:] = (u[:,  1 , :,:] - u[:,  0  , :,:]) / h[0]
        du[:, -1  , :,:] = (u[:, -1 , :,:] - u[:, -2  , :,:]) / h[0]
        return du


    @staticmethod
    def du_dy_central(u, h):
        du = torch.zeros_like(u)
        du[:,:, 1:-1, :] = (u[:,:,  2:, :] - u[:,:, 0:-2, :]) / (2 * h[1])
        du[:,:,  0  , :] = (u[:,:,  1 , :] - u[:,:,  0  , :]) / h[1]
        du[:,:, -1  , :] = (u[:,:, -1 , :] - u[:,:, -2  , :]) / h[1]
        return du


    @staticmethod
    def du_dz_central(u, h):
        du = torch.zeros_like(u)
        du[:,:,:, 1:-1] = (u[:,:,:,  2:] - u[:,:,:, 0:-2]) / (2 * h[2])
        du[:,:,:,  0  ] = (u[:,:,:,  1 ] - u[:,:,:, 0  ]) / h[2]
        du[:,:,:, -1  ] = (u[:,:,:, -1 ] - u[:,:,:, -2  ]) / h[2]
        return du


    @staticmethod
    def du_dx_forward(u, h):
        du = torch.zeros_like(u)
        du[:, 0:-1,:,:] = (u[:,  1:,:,:] - u[:, 0:-1,:,:]) / h[0]
        du[:, -1  ,:,:] = (u[:, -1 ,:,:] - u[:, -2  ,:,:]) / h[0]
        return du


    @staticmethod
    def du_dy_forward(u, h):
        du = torch.zeros_like(u)
        du[:,:, 0:-1,:] = (u[:,:,  1:,:] - u[:,:, 0:-1,:]) / h[1]
        du[:,:, -1  ,:] = (u[:,:, -1 ,:] - u[:,:, -2  ,:]) / h[1]
        return du


    @staticmethod
    def du_dz_forward(u, h):
        du = torch.zeros_like(u)
        du[:,:,:, 0:-1] = (u[:,:,:,  1:] - u[:,:,:, 0:-1]) / h[2]
        du[:,:,:, -1  ] = (u[:,:,:, -1 ] - u[:,:,:, -2  ]) / h[2]
        return du


    @staticmethod
    def du_dx(u, h, use_forward_differences=True):
        assert len(u.shape) == 4
        assert u.shape[1] > 2
        if use_forward_differences:
            return FDMDerivatives.du_dx_forward(u, h)
        return FDMDerivatives.du_dx_central(u, h)


    @staticmethod
    def du_dy(u, h, use_forward_differences=True):
        assert len(u.shape) == 4
        assert u.shape[2] > 2
        if use_forward_differences:
            return FDMDerivatives.du_dy_forward(u, h)
        return FDMDerivatives.du_dy_central(u, h)


    @staticmethod
    def du_dz(u, h, use_forward_differences=True):
        assert len(u.shape) == 4
        assert u.shape[3] > 2
        if use_forward_differences:
            return FDMDerivatives.du_dz_forward(u, h)
        return FDMDerivatives.du_dz_central(u, h)

# Cell
class FDMAdjointDerivatives():
    @staticmethod
    def du_dx_adj_for_a_sufficiently_large_number_of_voxels(ε, h):
        u = torch.zeros_like(ε)
        u[:,   0,:,:] = -(2 * ε[:,   0,:,:] + ε[:,   1,:,:]) / (2 * h[0])
        u[:,   1,:,:] =  (2 * ε[:,   0,:,:] - ε[:,   2,:,:]) / (2 * h[0])
        u[:,2:-2,:,:] =  (    ε[:,1:-3,:,:] - ε[:,3:-1,:,:]) / (2 * h[0])
        u[:,  -2,:,:] = -(2 * ε[:,  -1,:,:] - ε[:,  -3,:,:]) / (2 * h[0])
        u[:,  -1,:,:] =  (2 * ε[:,  -1,:,:] + ε[:,  -2,:,:]) / (2 * h[0])
        return u


    @staticmethod
    def du_dy_adj_for_a_sufficiently_large_number_of_voxels(ε, h):
        u = torch.zeros_like(ε)
        u[:,:,   0,:] = -(2 * ε[:,:,   0,:] + ε[:,:,   1,:]) / (2 * h[1])
        u[:,:,   1,:] =  (2 * ε[:,:,   0,:] - ε[:,:,   2,:]) / (2 * h[1])
        u[:,:,2:-2,:] =  (    ε[:,:,1:-3,:] - ε[:,:,3:-1,:]) / (2 * h[1])
        u[:,:,  -2,:] = -(2 * ε[:,:,  -1,:] - ε[:,:,  -3,:]) / (2 * h[1])
        u[:,:,  -1,:] =  (2 * ε[:,:,  -1,:] + ε[:,:,  -2,:]) / (2 * h[1])
        return u


    @staticmethod
    def du_dz_adj_for_a_sufficiently_large_number_of_voxels(ε, h):
        u = torch.zeros_like(ε)
        u[:,:,:,   0] = -(2 * ε[:,:,:,   0] + ε[:,:,:,   1]) / (2 * h[2])
        u[:,:,:,   1] =  (2 * ε[:,:,:,   0] - ε[:,:,:,   2]) / (2 * h[2])
        u[:,:,:,2:-2] =  (    ε[:,:,:,1:-3] - ε[:,:,:,3:-1]) / (2 * h[2])
        u[:,:,:,  -2] = -(2 * ε[:,:,:,  -1] - ε[:,:,:,  -3]) / (2 * h[2])
        u[:,:,:,  -1] =  (2 * ε[:,:,:,  -1] + ε[:,:,:,  -2]) / (2 * h[2])
        return u


    @staticmethod
    def du_dx_adj_for_a_sufficiently_small_number_of_voxels(ε, h):
        u = torch.zeros_like(ε)

        if u.shape[1] == 2:
            u[:, 0,:,:] = -(ε[:, 0,:,:] +  ε[:, 1,:,:]) / h[0]
            u[:, 1,:,:] = - u[:, 0,:,:]

        if u.shape[1] == 3:
            u[:,0,:,:] = -(2 * ε[:,0,:,:] + ε[:,1,:,:]) / (2 * h[0])
            u[:,1,:,:] =  (    ε[:,0,:,:] - ε[:,2,:,:]) /  h[0]
            u[:,2,:,:] =  (2 * ε[:,2,:,:] + ε[:,1,:,:]) / (2 * h[0])

        return u


    @staticmethod
    def du_dy_adj_for_a_sufficiently_small_number_of_voxels(ε, h):
        u = torch.zeros_like(ε)

        if u.shape[2] == 2:
            u[:,:, 0,:] = -(ε[:,:, 0,:] +  ε[:,:, 1,:]) / h[1]
            u[:,:, 1,:] = - u[:,:, 0,:]

        if u.shape[2] == 3:
            u[:,:,0,:] = -(2 * ε[:,:,0,:] + ε[:,:,1,:]) / (2 * h[1])
            u[:,:,1,:] =  (    ε[:,:,0,:] - ε[:,:,2,:]) /  h[1]
            u[:,:,2,:] =  (2 * ε[:,:,2,:] + ε[:,:,1,:]) / (2 * h[1])

        return u


    @staticmethod
    def du_dz_adj_for_a_sufficiently_small_number_of_voxels(ε, h):
        u = torch.zeros_like(ε)

        if u.shape[3] == 2:
            u[:,:,:, 0] = -(ε[:,:,:, 0] +  ε[:,:,:, 1]) / h[2]
            u[:,:,:, 1] = - u[:,:,:, 0]

        if u.shape[3] == 3:
            u[:,:,:,0] = -(2 * ε[:,:,:,0] + ε[:,:,:,1]) / (2 * h[2])
            u[:,:,:,1] =  (    ε[:,:,:,0] - ε[:,:,:,2]) /  h[2]
            u[:,:,:,2] =  (2 * ε[:,:,:,2] + ε[:,:,:,1]) / (2 * h[2])

        return u


    @staticmethod
    def du_dx_adj_forward(ε, h):
        u = torch.zeros_like(ε)
        u[:,   0   ,:,:] =  (                 - ε[:,      0,:,:]) / h[0]
        u[:,   1:-2,:,:] =  (ε[:,   0:-3,:,:] - ε[:,   1:-2,:,:]) / h[0]
        u[:,     -2,:,:] =  (ε[:,  -3,:,:] - ε[:,  -2,:,:] - ε[:,  -1,:,:]) / h[0]
        u[:,     -1,:,:] =  (ε[:,     -2,:,:] + ε[:,     -1,:,:]) / h[0]
        return u


    @staticmethod
    def du_dy_adj_forward(ε, h):
        u = torch.zeros_like(ε)
        u[:,:,   0   ,:] =  (                 - ε[:,:,      0,:]) / h[1]
        u[:,:,   1:-2,:] =  (ε[:,:,   0:-3,:] - ε[:,:,   1:-2,:]) / h[1]
        u[:,:,     -2,:] =  (ε[:,:,  -3,:] - ε[:,:,  -2,:] - ε[:,:,  -1,:]) / h[1]
        u[:,:,     -1,:] =  (ε[:,:,     -2,:] + ε[:,:,     -1,:]) / h[1]
        return u


    @staticmethod
    def du_dz_adj_forward(ε, h):
        u = torch.zeros_like(ε)
        u[:,:,:,   0   ] =  (                 - ε[:,:,:,      0]) / h[2]
        u[:,:,:,   1:-2] =  (ε[:,:,:,   0:-3] - ε[:,:,:,   1:-2]) / h[2]
        u[:,:,:,     -2] =  (ε[:,:,:,  -3] - ε[:,:,:,  -2] - ε[:,:,:,  -1]) / h[2]
        u[:,:,:,     -1] =  (ε[:,:,:,     -2] + ε[:,:,:,     -1]) / h[2]
        return u


    @staticmethod
    def du_dx_adj(ε, h, use_forward_differences=True):
        assert len(ε.shape) == 4
        if use_forward_differences:
            return FDMAdjointDerivatives.du_dx_adj_forward(ε, h)


        if ε.shape[1] > 3:
            return FDMAdjointDerivatives.du_dx_adj_for_a_sufficiently_large_number_of_voxels(ε, h)
        return FDMAdjointDerivatives.du_dx_adj_for_a_sufficiently_small_number_of_voxels(ε, h)


    @staticmethod
    def du_dy_adj(ε, h, use_forward_differences=True):
        assert len(ε.shape) == 4
        if use_forward_differences:
            return FDMAdjointDerivatives.du_dy_adj_forward(ε, h)


        if ε.shape[2] > 3:
            return FDMAdjointDerivatives.du_dy_adj_for_a_sufficiently_large_number_of_voxels(ε, h)
        return FDMAdjointDerivatives.du_dy_adj_for_a_sufficiently_small_number_of_voxels(ε, h)


    @staticmethod
    def du_dz_adj(ε, h, use_forward_differences=True):
        assert len(ε.shape) == 4
        if use_forward_differences:
            return FDMAdjointDerivatives.du_dz_adj_forward(ε, h)


        if ε.shape[3] > 3:
            return FDMAdjointDerivatives.du_dz_adj_for_a_sufficiently_large_number_of_voxels(ε, h)
        return FDMAdjointDerivatives.du_dz_adj_for_a_sufficiently_small_number_of_voxels(ε, h)

# Internal Cell
import torch
import numpy as np
from scipy.sparse import csc_matrix, hstack

# Cell
class FDMAssembly():
    """
    This class contains methods that are used for the assembly of the FDM stiffness matrix.
    """

    @staticmethod
    def apply_dirichlet_zero_rows_to_operator(operator, Ω_dirichlet):
        """
        Returns a version of `operator` that fulfills the dirichlet conditions in the output.

        Returns
        -------
        torch.Tensor
        """
        def operator_with_dirichlet_rows_zero(x):
            assert len(Ω_dirichlet.shape) == len(x.shape) == 4
            y = operator(x)
            y[Ω_dirichlet] = 0
            return y

        return operator_with_dirichlet_rows_zero


    @staticmethod
    def apply_dirichlet_zero_columns_to_operator(operator, Ω_dirichlet):
        """
        Returns a version of `operator` that fulfills the dirichlet conditions in the input.

        Returns
        -------
        torch.Tensor
        """
        def operator_with_dirichlet_columns_zero(x):
            assert len(Ω_dirichlet.shape) == len(x.shape) == 4
            x = x.clone()
            x[Ω_dirichlet] = 0
            y = operator(x)
            return y

        return operator_with_dirichlet_columns_zero


    @staticmethod
    def _get_graph(operator, shape, channels_in, filter_shape):
        operator_graph = []

        for i in range(filter_shape[0]):
            for j in range(filter_shape[1]):
                for k in range(filter_shape[2]):
                    for c in range(channels_in):
                        x = torch.zeros(channels_in, *shape)
                        x[c, i::filter_shape[0], j::filter_shape[1], k::filter_shape[2]] = 1
                        operator_graph.append((x.numpy(), operator(x).numpy()))

        return operator_graph


    @staticmethod
    def _get_nbh_coordinates(pos_in_nbhs, channels_in, channels_out):
        assert pos_in_nbhs.shape[1:] == (4,)
        channels_prod = channels_in * channels_out
        nbh_coordinates = -2 * np.ones([pos_in_nbhs.shape[0], channels_prod*channels_out, 4])
        centroids = np.zeros([pos_in_nbhs.shape[0], channels_prod*channels_out, 4])

        if pos_in_nbhs.shape[0] > 0:
            centroids[:] = pos_in_nbhs[0]

        for c in range(channels_out):
            nbh_coordinates[:, channels_prod*c:channels_prod*(c+1), 0] = c

        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if [i,j,k].count(0) >= 2:
                        rhs = pos_in_nbhs[:, 1:] + np.array([i, j, k])
                        t = channels_out * (i + 1) + channels_in * (j + 1) + (k + 1)
                        nbh_coordinates[:, t::channels_prod, 1:] = rhs.reshape(pos_in_nbhs.shape[0], 1, 3)
                        centroids[:, t::channels_prod, 1:] = pos_in_nbhs[:, 1:].reshape(pos_in_nbhs.shape[0], 1, 3)

        assert nbh_coordinates.shape[1:] == (channels_prod*channels_out, 4)
        return nbh_coordinates.reshape(-1, 4), centroids.reshape(-1, 4)


    @staticmethod
    def _remove_out_of_bounds_rows(nbh_coordinates, centroids, shape):
        mask0 = np.all(nbh_coordinates >=0, axis=1, keepdims=True).flatten()
        mask1 = nbh_coordinates[:,1] < shape[0]
        mask2 = nbh_coordinates[:,2] < shape[1]
        mask3 = nbh_coordinates[:,3] < shape[2]

        mask = mask0 & mask1 & mask2 & mask3
        return nbh_coordinates[mask].astype(int), centroids[mask].astype(int)


    @staticmethod
    def _get_1d_coordinates(positions, shape):
        assert positions.shape[1:] == (4,)
        coords_1d = positions[:,3] + shape[2]*positions[:,2] + shape[1] * shape[2] * positions[:,1] +  shape[0] * shape[1] * shape[2] * positions[:,0]
        return coords_1d.astype(int)


    @staticmethod
    def assemble_operator(operator, shape, channels_in=3, channels_out=9, filter_shape=3, Ω_dirichlet=None, column_wise=True):
        """
        Returns a sparse assembly of `operator`.

        Returns
        -------
        scipy.sparse.csc_matrix
        """
        if type(filter_shape) is int:
            filter_shape = [filter_shape, filter_shape, filter_shape]

        if Ω_dirichlet is not None:
            if column_wise:
                operator = FDMAssembly.apply_dirichlet_zero_columns_to_operator(operator, Ω_dirichlet)
            else:
                operator = FDMAssembly.apply_dirichlet_zero_rows_to_operator(operator, Ω_dirichlet)

        op_graph = FDMAssembly._get_graph(operator, shape, channels_in, filter_shape)
        col_indices = []
        row_indices = []
        values = []

        for x, y in op_graph:
            pos_in_nbhs = np.where(x)
            pos_in_nbhs = np.stack(pos_in_nbhs).transpose()

            nbh_3d, centroids = FDMAssembly._get_nbh_coordinates(pos_in_nbhs, channels_in, channels_out)
            nbh_3d, centroids = FDMAssembly._remove_out_of_bounds_rows(nbh_3d, centroids, shape)

            col_idx = FDMAssembly._get_1d_coordinates(centroids, shape)
            row_idx = FDMAssembly._get_1d_coordinates(nbh_3d, shape)

            col_indices.extend(col_idx)
            row_indices.extend(row_idx)

            vals = np.take(y.flatten(), row_idx, axis=0)
            values.extend(vals)

        return csc_matrix((values, (row_indices, col_indices)), shape=(channels_out*np.prod(shape), channels_in*np.prod(shape)))

# Internal Cell
import torch
import warnings
import numpy as np
from scipy.sparse import diags, csc_matrix


# from .pde import SparseLinearSolver, PDESolver, FDMDerivatives, FDMAdjointDerivatives, FDMAssembly
from .utils import get_σ_vm

# Cell
class UnpaddedFDM(PDESolver):
    """
    A PDE solver for linear elasticity that uses the finite differences method (FDM) with padding.
    """
    def __init__(self, θ_min:float=1e-6, # The minimal value in the stiffness matrix. For numerical reasons we can not allow 0s, since they may lead to singular matrices.
                use_forward_differences:bool=True, # Whether to use forward differences or central differences.
                assemble_tensors_when_passed_to_problem:bool=True, # Whether the PDE solver methods pre-assembles any tensors or arrays before solving the PDE for a concrete problem.
                interpolation_model:str='simp', # 'simp' or 'ramp'
                ramp_q:float=8.0 # RAMP parameter q controlling penalization strength
                ):
        self._θ_min = θ_min
        # Use iterative solver (factorize=False) to allow θ-based warm-start gating & GPU path
        # Use a conservative default backend (scipy). User can override later.
        self._linear_solver = SparseLinearSolver(
            factorize=False,
            optimizer="scipy",
            preconditioner="jacobi"
        )
        self.use_forward_differences = use_forward_differences
        self.assemble_tensors_when_passed_to_problem = assemble_tensors_when_passed_to_problem
        self.assembled_tensors = False
        self.interpolation_model = interpolation_model.lower()
        assert self.interpolation_model in ('simp', 'ramp'), "interpolation_model must be 'simp' or 'ramp'"
        self.ramp_q = float(ramp_q)
        super().__init__(assemble_tensors_when_passed_to_problem)


    @property
    def problem(self):
        return self._problem


    @property
    def shape(self):
        return self.problem.shape


    @property
    def Ω_dirichlet(self):
        return self.problem.Ω_dirichlet


    @property
    def θ_min(self):
        return self._θ_min


    @property
    def b(self):
        return self._b


    @property
    def h(self):
        return self.problem.h


    @property
    def linear_solver(self):
        return self._linear_solver


    def assemble_tensors(self,
                        problem # The problem for which the tensors should be assembled.
                        ):
        """
        Assembles all FDM tensors from the problem object that can be pre-built without knowledge of the density distribution `θ`. This may take some time but makes future PDE evaluations for this problem much faster.
        """
        self._problem = problem.clone()
        GJ = lambda u: self._G(self._J(u))
        self._Ω_dirichlet_diags = diags(self.Ω_dirichlet.flatten().int().numpy())
        self._Jt_mat = FDMAssembly.assemble_operator(
            operator=self._J, shape=self.shape,
            Ω_dirichlet=self.Ω_dirichlet,
            filter_shape=3).transpose()
        self._GJ_mat = FDMAssembly.assemble_operator(
            operator=GJ, shape=self.shape, Ω_dirichlet=self.Ω_dirichlet, filter_shape=3)
        self._b = self._get_b()
        self.assembled_tensors = True


    def _get_θ_from_solution(self, solution, binary=False, clone=False):
        if clone:
            θ = solution.get_θ(binary).clone()
        else:
            θ = solution.get_θ(binary)
        return θ


    def _J(self, u, dirichlet=False):
        J = lambda u: torch.cat([
            FDMDerivatives.du_dx(u, self.h, self.use_forward_differences),
            FDMDerivatives.du_dy(u, self.h, self.use_forward_differences),
            FDMDerivatives.du_dz(u, self.h, self.use_forward_differences)
        ], dim=0)

        if dirichlet:
            return FDMAssembly.apply_dirichlet_zero_columns_to_operator(J, self.Ω_dirichlet)(u)
        return J(u)


    def _J_adj(self, σ, dirichlet=False):
        Jt = lambda σ: FDMAdjointDerivatives.du_dx_adj(σ[:3],  self.h, self.use_forward_differences) + \
                    FDMAdjointDerivatives.du_dy_adj(σ[3:6], self.h, self.use_forward_differences) + \
                    FDMAdjointDerivatives.du_dz_adj(σ[6:],  self.h, self.use_forward_differences)

        if dirichlet:
            return FDMAssembly.apply_dirichlet_zero_rows_to_operator(Jt, self.Ω_dirichlet)(σ)
        return Jt(σ)


    def _get_G(self):
        ν = self.problem.ν

        G = torch.tensor([
            [1-ν,  0-0,  0-0,    0-0,  ν-0,  0-0,    0-0,  0-0,  ν-0],
            [0-0, .5-ν,  0-0,   .5-ν,  0-0,  0-0,    0-0,  0-0,  0-0],
            [0-0,  0-0, .5-ν,    0-0,  0-0,  0-0,   .5-ν,  0-0,  0-0],

            [0-0, .5-ν,  0-0,   .5-ν,  0-0,  0-0,    0-0,  0-0,  0-0],
            [ν-0,  0-0,  0-0,    0-0,  1-ν,  0-0,    0-0,  0-0,  ν-0],
            [0-0,  0-0,  0-0,    0-0,  0-0, .5-ν,    0-0, .5-ν,  0-0],

            [0-0,  0-0, .5-ν,    0-0,  0-0,  0-0,   .5-ν,  0-0,  0-0],
            [0-0,  0-0,  0-0,    0-0,  0-0, .5-ν,    0-0, .5-ν,  0-0],
            [ν-0,  0-0,  0-0,    0-0,  ν-0,  0-0,    0-0,  0-0,  1-ν]
        ], dtype=self.problem.dtype)

        G = G.to(self.problem.device)
        return G / ((1 + ν) * (1 - 2 * ν))


    def _G(self, ε):
        ε = ε.type(self.problem.dtype)
        return torch.einsum('ij, jlmn -> ilmn', self._get_G(), ε)


    def _G_adj(self, σ):
        σ = σ.type(self.problem.dtype)
        return torch.einsum('ij, jlmn -> ilmn', self._get_G().t(), σ)


    def _GJ(self, u, dirichlet=False):
        apply_GJ = lambda u: self._G(self._J(u))

        if dirichlet:
            return FDMAssembly.apply_dirichlet_zero_columns_to_operator(apply_GJ, self.Ω_dirichlet)(u)
        return apply_GJ(u)


    def _GJ_adj(self, σ, dirichlet=False):
        apply_GJ_adj = lambda σ: self._J_adj(self._G_adj(σ))

        if dirichlet:
            return FDMAssembly.apply_dirichlet_zero_rows_to_operator(apply_GJ_adj, self.Ω_dirichlet)(σ)
        return apply_GJ_adj(σ)


    def _apply_θp(self, σ, θ, p=1., normalize=True):
        E = 1. if normalize else self.problem.E
        E_min = E * self.θ_min
        frac = self._interp_fraction(θ, p)
        θ_eff = E_min + frac * (E - E_min)
        return θ_eff * σ


    def _assemble_θ(self, θ, p=1.):
        E = 1.
        E_min = E * self.θ_min
        frac = self._interp_fraction(θ, p)
        θ_flat = (E_min + frac * (E - E_min)).flatten().repeat(9).detach().numpy()
        return diags(θ_flat)

    def _interp_fraction(self, θ: torch.Tensor, p: float):
        """Return dimensionless interpolation fraction in [θ_min, 1] for the chosen model.

        For SIMP: frac = θ**p
        For RAMP: frac = θ / (1 + q * (1 - θ))
        """
        if self.interpolation_model == 'simp':
            return θ.clamp(self.θ_min, 1.0) ** p
        # RAMP
        q = self.ramp_q
        θc = θ.clamp(self.θ_min, 1.0)
        return θc / (1.0 + q * (1.0 - θc))


    def _A(self, u, θ, dirichlet=True, p=1.):
        u = u.view(3, θ.shape[-3], θ.shape[-2], θ.shape[-1])
        y = self._GJ(u, dirichlet)
        y = self._apply_θp(y, θ, p)
        y = self._J_adj(y, dirichlet)

        if dirichlet:
            y[self.Ω_dirichlet] = u.clone()[self.Ω_dirichlet]
        return y


    def _A_adj(self, y, θ, dirichlet=True, p=1.):
        y = y.view(3, θ.shape[-3], θ.shape[-2], θ.shape[-1])
        u = self._J(y, dirichlet)
        u = self._apply_θp(u, θ, p)
        u = self._GJ_adj(u, dirichlet)

        if dirichlet:
            u[self.Ω_dirichlet] = y.clone()[self.Ω_dirichlet]
        return u


    def _assemble_A(self, θ, p=1.):
        # Optimized assemble: in-place row scaling of base GJ (avoid building diag matrix)
        # θ_eff repeats per strain component (9 per voxel) to match GJ row blocks.
        E = 1.0
        E_min = E * self.θ_min
        frac = self._interp_fraction(θ, p)
        θ_eff_np = (E_min + frac * (E - E_min)).flatten().repeat(9).detach().numpy()

        base = self._GJ_mat.tocsr().copy()
        base.sort_indices()
        base = csr_matrix((base.data.astype(np.float64, copy=False),
                           base.indices.astype(np.int32, copy=False),
                           base.indptr.astype(np.int32, copy=False)), shape=base.shape)
        GJ = base.copy()

        row_ids = np.empty(GJ.data.shape[0], dtype=np.int64)
        for r in range(GJ.shape[0]):
            start, end = GJ.indptr[r], GJ.indptr[r + 1]
            if start != end:
                row_ids[start:end] = r
        np.multiply(GJ.data, θ_eff_np[row_ids], out=GJ.data)

        Jt_csr = self._Jt_mat.tocsr().copy()
        Jt_csr.sort_indices()
        Jt_csr = csr_matrix((Jt_csr.data.astype(np.float64, copy=False),
                             Jt_csr.indices.astype(np.int32, copy=False),
                             Jt_csr.indptr.astype(np.int32, copy=False)), shape=Jt_csr.shape)

        verbose_gpu = os.getenv('PDE_ASSEMBLY_VERBOSE', '0') == '1'
        use_petsc_gpu = os.getenv('PDE_ASSEMBLY_USE_PETSC_GPU', '0') == '1'

        def _finalize_csr(A_csr_in: csr_matrix):
            A_local = A_csr_in
            if self._Ω_dirichlet_diags.nnz:
                A_local = A_local + self._Ω_dirichlet_diags.tocsr()
            A_local.sort_indices()
            A_local = csr_matrix((A_local.data.astype(np.float64, copy=False),
                                  A_local.indices.astype(np.int32, copy=False),
                                  A_local.indptr.astype(np.int32, copy=False)), shape=A_local.shape)
            A_csc_local = A_local.tocsc()
            A_csc_local.sort_indices()
            A_csc_local.indices = A_csc_local.indices.astype(np.int32, copy=False)
            A_csc_local.indptr = A_csc_local.indptr.astype(np.int32, copy=False)
            return A_csc_local

        def _petsc_matmul(Jt_cpu, GJ_cpu):

            print(f"[PETSc-DEBUG] Entering _petsc_matmul, Jt shape={Jt_cpu.shape}, GJ shape={GJ_cpu.shape}", flush=True)
            
            if not HAVE_PETSC4PY or PETSc is None or petsc4py is None:
                raise RuntimeError('petsc4py is not available')
            
            print(f"[PETSc-DEBUG] Checking PETSc initialization...", flush=True)
            if not PETSc.Sys.isInitialized():
                # Set CUDA device for PETSc before initialization
                petsc_cuda_device = os.getenv('PETSC_CUDA_DEVICE')
                if petsc_cuda_device is not None:
                    print(f"[PETSc-DEBUG] Setting CUDA device for PETSc to: {petsc_cuda_device}", flush=True)
                    # PETSc respects these environment variables
                    os.environ['CUDA_DEVICE'] = str(petsc_cuda_device)
                    os.environ['HYPRE_CUDA_DEVICE'] = str(petsc_cuda_device)
                    # For process isolation, set CUDA_VISIBLE_DEVICES
                    # NOTE: This will affect the entire process, so only set if explicitly requested
                    petsc_cuda_visible = os.getenv('PETSC_CUDA_VISIBLE_DEVICES')
                    if petsc_cuda_visible is not None:
                        print(f"[PETSc-DEBUG] Setting CUDA_VISIBLE_DEVICES for PETSc to: {petsc_cuda_visible}", flush=True)
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(petsc_cuda_visible)
                else:
                    print(f"[PETSc-DEBUG] No PETSC_CUDA_DEVICE specified, using default GPU", flush=True)
                
                print(f"[PETSc-DEBUG] Initializing PETSc...", flush=True)
                petsc4py.init([])
                print(f"[PETSc-DEBUG] PETSc initialized successfully", flush=True)
            else:
                print(f"[PETSc-DEBUG] PETSc already initialized", flush=True)

            mat_type_preference = os.getenv('PDE_ASSEMBLY_PETSC_TYPE', 'aijcusparse')
            print(f"[PETSc-DEBUG] Matrix type preference: {mat_type_preference}", flush=True)

            def _csr_to_petsc(csr_obj, name="matrix"):
                print(f"[PETSc-DEBUG] Converting {name} to PETSc format, shape={csr_obj.shape}, nnz={csr_obj.nnz}", flush=True)
                
                mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
                print(f"[PETSc-DEBUG] {name}: Created PETSc Mat object", flush=True)
                
                mat.setSizes(csr_obj.shape)
                print(f"[PETSc-DEBUG] {name}: Set sizes to {csr_obj.shape}", flush=True)
                
                type_candidates = []
                if mat_type_preference:
                    type_candidates.append(mat_type_preference)
                type_candidates.append('aij')
                
                mat_type_set = None
                for mat_type in type_candidates:
                    try:
                        print(f"[PETSc-DEBUG] {name}: Trying setType({mat_type})...", flush=True)
                        mat.setType(mat_type)
                        mat_type_set = mat_type
                        print(f"[PETSc-DEBUG] {name}: Successfully set type to {mat_type}", flush=True)
                        break
                    except Exception as type_err:
                        print(f"[PETSc-DEBUG] {name}: setType({mat_type}) failed: {type_err}", flush=True)
                        if verbose_gpu:
                            print(f"[PDE] PETSc setType({mat_type}) failed: {type_err}", flush=True)
                
                mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
                print(f"[PETSc-DEBUG] {name}: Set options", flush=True)
                
                mat.setUp()
                print(f"[PETSc-DEBUG] {name}: setUp() complete", flush=True)
                
                indptr = csr_obj.indptr.astype(np.int32, copy=False)
                indices = csr_obj.indices.astype(np.int32, copy=False)
                data = csr_obj.data.astype(np.float64, copy=False)
                print(f"[PETSc-DEBUG] {name}: Converted arrays - indptr len={len(indptr)}, indices len={len(indices)}, data len={len(data)}", flush=True)
                
                print(f"[PETSc-DEBUG] {name}: Calling setValuesCSR...", flush=True)
                mat.setValuesCSR(indptr, indices, data)
                print(f"[PETSc-DEBUG] {name}: setValuesCSR complete", flush=True)
                
                print(f"[PETSc-DEBUG] {name}: Calling assemblyBegin...", flush=True)
                mat.assemblyBegin()
                print(f"[PETSc-DEBUG] {name}: Calling assemblyEnd...", flush=True)
                mat.assemblyEnd()
                print(f"[PETSc-DEBUG] {name}: Assembly complete with type {mat_type_set}", flush=True)
                
                return mat

            petsc_Jt = None
            petsc_GJ = None
            petsc_A = None
            try:
                print(f"[PETSc-DEBUG] Converting Jt matrix...", flush=True)
                petsc_Jt = _csr_to_petsc(Jt_cpu, "Jt")
                
                print(f"[PETSc-DEBUG] Converting GJ matrix...", flush=True)
                petsc_GJ = _csr_to_petsc(GJ_cpu, "GJ")
                
                print(f"[PETSc-DEBUG] Starting matMult operation...", flush=True)
                petsc_A = petsc_Jt.matMult(petsc_GJ)
                print(f"[PETSc-DEBUG] matMult complete, assembling result...", flush=True)
                
                petsc_A.assemblyBegin()
                print(f"[PETSc-DEBUG] Result assemblyBegin complete", flush=True)
                
                petsc_A.assemblyEnd()
                print(f"[PETSc-DEBUG] Result assemblyEnd complete", flush=True)
                
                print(f"[PETSc-DEBUG] Extracting CSR values from result...", flush=True)
                indptr, indices, data = petsc_A.getValuesCSR()
                print(f"[PETSc-DEBUG] getValuesCSR complete, converting to numpy...", flush=True)
                
                data_np = np.asarray(data, dtype=np.float64).copy()
                indices_np = np.asarray(indices, dtype=np.int32).copy()
                indptr_np = np.asarray(indptr, dtype=np.int32).copy()
                shape = petsc_A.getSize()
                print(f"[PETSc-DEBUG] Numpy conversion complete, result shape={shape}", flush=True)
                
                print(f"[PETSc-DEBUG] Creating scipy CSR matrix...", flush=True)
                result = csr_matrix((data_np, indices_np, indptr_np), shape=shape)
                print(f"[PETSc-DEBUG] _petsc_matmul completed successfully", flush=True)
                return result
            finally:
                print(f"[PETSc-DEBUG] Cleaning up PETSc matrices...", flush=True)
                for mat_name, mat in [("Jt", petsc_Jt), ("GJ", petsc_GJ), ("A", petsc_A)]:
                    try:
                        if mat is not None:
                            print(f"[PETSc-DEBUG] Destroying {mat_name}...", flush=True)
                            mat.destroy()
                            print(f"[PETSc-DEBUG] {mat_name} destroyed", flush=True)
                    except Exception as e:
                        print(f"[PETSc-DEBUG] Error destroying {mat_name}: {e}", flush=True)

        if use_petsc_gpu:
            if not HAVE_PETSC4PY:
                if verbose_gpu:
                    print('[PDE] PETSc GPU path requested but petsc4py is unavailable; falling back to CPU assembly.', flush=True)
            else:
                try:
                    A_csr_petsc = _petsc_matmul(Jt_csr, GJ)
                    if verbose_gpu:
                        print('[PDE] PETSc MatMatMult completed successfully.', flush=True)
                    return _finalize_csr(A_csr_petsc)
                except Exception as pet_err:
                    if verbose_gpu:
                        print(f'[PDE] PETSc MatMatMult failed ({pet_err}); reverting to existing path.', flush=True)

        # CPU fallback
        A_csr = Jt_csr.dot(GJ)
        return _finalize_csr(A_csr)


    def _get_b(self):
        b = self.problem.F
        b = self._get_padded_tensor(b)
        b[self.Ω_dirichlet] = 0
        b /= self.problem.E
        return b


    def _get_u(self, solution, p=1., binary=False):
        if binary and (solution.u_binary is not None):
            return solution.u_binary

        if (not binary) and (solution.u is not None):
            return solution.u

        if not self.assembled_tensors:
            self.assemble_tensors(solution.problem)

        θ = self._get_θ_from_solution(solution, binary=binary, clone=True)
        θ = θ.clamp(self.θ_min, 1)
        A_op = lambda u, θ: self._A(u, θ, p=p)
        t_asm0 = _pde_now()
        A_mat = self._assemble_A(θ.cpu(), p)
        _pde_log("assemble_A", t_asm0, extra=f"nnz={getattr(A_mat,'nnz', 'NA')}")
        t_solv0 = _pde_now()
        u = self._linear_solver(θ.cpu(), A_op, self.b.flatten(), A_mat)
        _pde_log("linear_solve", t_solv0, extra=f"shape={A_mat.shape}")
        u = u.view(3, θ.shape[-3], θ.shape[-2], θ.shape[-1]).to(θ.device)

        if binary:
            solution.u_binary = u.clone()
        else:
            solution.u = u.clone()

        return u


    def _get_σ(self, solution, p=1., u=None, binary=False):
        if u is None:
            u = self._get_u(solution, p=p, binary=binary)
        t_strain0 = _pde_now()
        θ = self._get_θ_from_solution(solution, binary=binary, clone=False)
        ε = self._J(u)
        σ = self._G(ε)
        σ = self._apply_θp(σ, θ, p=1., normalize=False)
        _pde_log("stress_compute", t_strain0)
        return σ

    def _build_torchfem_rbm_B(self) -> torch.Tensor | None:
        """Construct 6 rigid-body modes (3 translations + 3 rotations) for a solid grid.

        Returns a dense torch tensor B of shape (ndof, m) in float64, matching
        the solver's channel-major DOF ordering, with Dirichlet DOFs zeroed out.
        If construction fails, returns None.
        """
        try:
            # Shapes
            _, X, Y, Z = self.Ω_dirichlet.shape  # (3, X, Y, Z)
            device = self.problem.device if hasattr(self.problem, 'device') else 'cpu'
            dtype = torch.float64

            # Coordinates centered around origin, scaled by spacing h
            hx, hy, hz = float(self.h[0]), float(self.h[1]), float(self.h[2])
            xs = (torch.arange(X, dtype=dtype) - 0.5 * (X - 1)) * hx
            ys = (torch.arange(Y, dtype=dtype) - 0.5 * (Y - 1)) * hy
            zs = (torch.arange(Z, dtype=dtype) - 0.5 * (Z - 1)) * hz
            Xg, Yg, Zg = torch.meshgrid(xs, ys, zs, indexing='ij')

            # Allocate 6 modes in (3, X, Y, Z)
            modes = torch.zeros((6, 3, X, Y, Z), dtype=dtype)
            # Translations
            modes[0, 0, :, :, :] = 1.0  # tx
            modes[1, 1, :, :, :] = 1.0  # ty
            modes[2, 2, :, :, :] = 1.0  # tz
            # Rotations (u = ω x r)
            # About x: u = (0, -z, y)
            modes[3, 1, :, :, :] = -Zg
            modes[3, 2, :, :, :] =  Yg
            # About y: u = (z, 0, -x)
            modes[4, 0, :, :, :] =  Zg
            modes[4, 2, :, :, :] = -Xg
            # About z: u = (-y, x, 0)
            modes[5, 0, :, :, :] = -Yg
            modes[5, 1, :, :, :] =  Xg

            # Zero out constrained DOFs
            Ωd = self.Ω_dirichlet
            for k in range(6):
                mk = modes[k]
                mk[Ωd] = 0.0

            # Flatten each mode in channel-major to match A,b flattening
            # Stack into (ndof, 6)
            cols = []
            for k in range(6):
                cols.append(modes[k].flatten())
            B = torch.stack(cols, dim=1)  # (ndof, 6)
            return B.to(dtype=dtype, device=device)
        except Exception:
            return None


    def solve_pde(self,
                 solution: Any, # The solution for which the PDE should be solved.
                 p: float = 1., # The SIMP exponent when solving the PDE. Should usually be left at its default value of `1.`.
                 binary: bool = False, # Whether the densities in the solution should be binarized before solving the PDE.
                 get_padded: bool = False # Unused for UnpaddedFDM; kept for API compatibility.
                 ):
        """
        Solves the pde for `solution` and SIMP exponent `p`. Returns three `torch.Tensor` objects: displacements `u`, stresses `σ` and von Mises stresses `σ_vm`.
        """
        u = self._get_u(solution, p=p, binary=binary)
        σ = self._get_σ(solution, p=p, u=u, binary=binary)
        σ_vm = get_σ_vm(σ)
        return u, σ, σ_vm
    

    # Cell
class FDM(UnpaddedFDM):
    """
    A PDE solver for linear elasticity that uses the finite differences method (FDM) with padding.
    """
    def __init__(self, θ_min:float=1e-6, # The minimal value in the stiffness matrix. For numerical reasons we can not allow 0s, since they may lead to singular matrices.
                use_forward_differences:bool=True, # Whether to use forward differences or central differences.
                assemble_tensors_when_passed_to_problem:bool=True, # Whether the PDE solver methods pre-assembles any tensors or arrays before solving the PDE for a concrete problem.
                padding_depth:int=0, # The depth of the padding surrounding the design space. In some cases, it is recommended to increase the padding depth to 2 to improve results but also increase running time.
                interpolation_model:str='simp', # 'simp' or 'ramp'
                ramp_q:float=8.0 # RAMP parameter q controlling penalization strength
                ):
        self.padding_depth = padding_depth
        super().__init__(
            θ_min=θ_min,
            use_forward_differences=use_forward_differences,
            assemble_tensors_when_passed_to_problem=assemble_tensors_when_passed_to_problem,
            interpolation_model=interpolation_model,
            ramp_q=ramp_q
        )


    @property
    def shape(self):
        return self.Ω_dirichlet.shape[-3:]


    @property
    def Ω_dirichlet(self):
        return self._get_padded_tensor(self.problem.Ω_dirichlet)


    def _get_padded_tensor(self, tensor):
        p_d = int(self.padding_depth)
        if p_d == 0:
            return tensor

        shape = tensor.shape
        assert len(shape) == 4
        padded_tensor = torch.zeros(
            shape[0], shape[1]+2*p_d, shape[2]+2*p_d, shape[3]+2*p_d, dtype=tensor.dtype
        )
        padded_tensor[:, p_d:-p_d, p_d:-p_d, p_d:-p_d] = tensor
        return padded_tensor


    def _remove_padding(self, tensor):
        p_d = int(self.padding_depth)
        if p_d == 0:
            return tensor

        shape = tensor.shape
        assert len(shape) == 4
        return tensor[:, p_d:-p_d, p_d:-p_d, p_d:-p_d]


    def _get_θ_from_solution(self, solution, binary=False, clone=False):
        θ = super()._get_θ_from_solution(solution, binary=binary, clone=clone)
        return self._get_padded_tensor(θ)


    def _A(self, u, θ, dirichlet=True, p=1.):
        if θ.shape[-3:] == self.shape[-3:]:
            return super()._A(u=u, θ=θ, dirichlet=dirichlet, p=p)
        θ_padded = self._get_padded_tensor(θ)
        assert θ_padded.shape[-3:] == self.shape[-3:]
        return super()._A(u=u, θ=θ_padded, dirichlet=dirichlet, p=p)


    def _A_adj(self, y, θ, dirichlet=True, p=1.):
        if θ.shape[-3:] == self.shape[-3:]:
            return super()._A_adj(y=y, θ=θ, dirichlet=dirichlet, p=p)
        θ_padded = self._get_padded_tensor(θ)
        assert θ_padded.shape[-3:] == self.shape[-3:]
        return super()._A_adj(y=y, θ=θ_padded, dirichlet=dirichlet, p=p)


    def _get_b(self):
        b = self.problem.F
        b = self._get_padded_tensor(b)
        b[self.Ω_dirichlet] = 0
        b /= self.problem.E
        return b


    def _get_u(self, solution, p=1., binary=False, get_padded=False):
        u = super()._get_u(solution, p=p, binary=binary)
        if get_padded:
            return u
        return self._remove_padding(u)


    def _get_σ(self, solution, p=1., u=None, binary=False, get_padded=False):
        if u is None:
            u = self._get_u(solution, p=p, binary=binary, get_padded=True)
        σ = super()._get_σ(solution, p=p, u=u, binary=binary)
        if get_padded:
            return σ
        return self._remove_padding(σ)


    def solve_pde(self,
                solution: Any, # The solution for which the PDE should be solved.
                p:float=1., # The SIMP exponent when solving the PDE. Should usually be left at its default value of `1.`.
                binary:bool=False, # Whether the densities in the solution should be binarized before solving the PDE.
                get_padded:bool=False # Whether the density should be padded before the PDE is solved. Takes a bit longer to solve, but is more accurate.
                ):
        """
        Solves the pde for `solution` and SIMP exponent `p`. Returns three `torch.Tensor` objects: displacements `u`, stresses `σ` and von Mises stresses `σ_vm`.
        """
        u = self._get_u(solution, p=p, binary=binary, get_padded=True)
        σ = self._get_σ(solution, p=p, u=u, binary=binary, get_padded=True)
        σ_vm = get_σ_vm(σ)
        if get_padded:
            return u, σ, σ_vm
        return self._remove_padding(u), self._remove_padding(σ), self._remove_padding(σ_vm)

import time as _pde_time_mod

_PDE_TIMING = os.getenv("PDE_TIMING", "0") == "1"

def _pde_now():
    return _pde_time_mod.perf_counter()

def _pde_log(stage: str, t_start: float, extra: str = ""):
    if _PDE_TIMING:
        dt = (_pde_time_mod.perf_counter() - t_start) * 1000.0
        msg = f"[PDE-TIME] {stage} {dt:.2f} ms"
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)