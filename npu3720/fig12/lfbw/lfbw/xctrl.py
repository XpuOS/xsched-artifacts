import ctypes
from enum import IntEnum

class ZePriority(IntEnum):
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL        = 0,
    ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW  = 1,
    ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH = 2,


class XSched:
    try:
        shim_dll = ctypes.cdll.LoadLibrary("libshimze.so")

        shim_dll.ZeXQueueCreate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
        shim_dll.ZeXQueueCreate.restype = ctypes.c_uint64

        shim_dll.ZeXQueueDestroy.argtypes = [ctypes.c_uint64]
        shim_dll.ZeXQueueDestroy.restype = None

        shim_dll.ZeXQueueSuspend.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        shim_dll.ZeXQueueSuspend.restype = None

        shim_dll.ZeXQueueResume.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        shim_dll.ZeXQueueResume.restype = None

        shim_dll.ZeSetOverrideHalQueue.argtypes = [ctypes.c_int]
        shim_dll.ZeSetOverrideHalQueue.restype = None

        shim_dll.ZeUnsetOverrideHalQueue.argtypes = []
        shim_dll.ZeUnsetOverrideHalQueue.restype = None

        shim_dll.ZeGetOverrideHalQueue.argtypes = [ctypes.c_int]
        shim_dll.ZeGetOverrideHalQueue.restype = ctypes.c_void_p

        shim_dll.ZeXQueueSetPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        shim_dll.ZeXQueueSetPriority.restype = None

        shim_dll.ZeXQueueSetBandwidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
        shim_dll.ZeXQueueSetBandwidth.restype = None

        shim_dll.ZeXQueueSetDeadline.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        shim_dll.ZeXQueueSetDeadline.restype = None

        shim_dll.ZeXQueueSetLaxity.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32]
        shim_dll.ZeXQueueSetLaxity.restype = None

        shim_dll.zeCommandQueueSynchronize.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        shim_dll.zeCommandQueueSynchronize.restype = ctypes.c_int

    except Exception as e:
        print(e)
        print("Error: libshimze.so not found, please add it to your LD_LIBRARY_PATH")
        exit(1)

    @staticmethod
    def xqueue_create(cmdq: int, preempt_mode: int, queue_length: int, sync_interval: int) -> int:
        return int(XSched.shim_dll.ZeXQueueCreate(ctypes.c_void_p(cmdq), preempt_mode, queue_length, sync_interval))
    
    @staticmethod
    def xqueue_destroy(xqueue_handle: int) -> None:
        XSched.shim_dll.ZeXQueueDestroy(xqueue_handle)
    
    @staticmethod
    def xqueue_suspend(cmdq: int, sync_hal_queue: bool) -> None:
        XSched.shim_dll.ZeXQueueSuspend(ctypes.c_void_p(cmdq), sync_hal_queue)
    
    @staticmethod   
    def xqueue_resume(cmdq: int, drop_commands: bool) -> None:
        XSched.shim_dll.ZeXQueueResume(ctypes.c_void_p(cmdq), drop_commands)
    
    @staticmethod
    def set_override_hal_queue(queue_idx: int) -> None:
        XSched.shim_dll.ZeSetOverrideHalQueue(queue_idx)
    
    @staticmethod
    def unset_override_hal_queue() -> None:
        XSched.shim_dll.ZeUnsetOverrideHalQueue()
    
    @staticmethod
    def get_override_hal_queue(queue_idx: int) -> int:
        return int(XSched.shim_dll.ZeGetOverrideHalQueue(queue_idx))
    
    @staticmethod
    def xqueue_set_priority(cmdq: int, prio: ZePriority) -> None:
        XSched.shim_dll.ZeXQueueSetPriority(ctypes.c_void_p(cmdq), int(prio))
    
    @staticmethod
    def xqueue_set_bandwidth(cmdq: int, bdw: int) -> None:
        XSched.shim_dll.ZeXQueueSetBandwidth(ctypes.c_void_p(cmdq), bdw)
    
    @staticmethod
    def xqueue_set_deadline(cmdq: int, deadline_us: int) -> None:
        XSched.shim_dll.ZeXQueueSetDeadline(ctypes.c_void_p(cmdq), deadline_us)

    @staticmethod
    def xqueue_set_laxity(cmdq: int, laxity_us: int, lax_prio: int, critical_prio: int) -> None:
        XSched.shim_dll.ZeXQueueSetLaxity(ctypes.c_void_p(cmdq), laxity_us, lax_prio, critical_prio)

    @staticmethod
    def ze_command_queue_synchronize(cmdq: int, timeout: int) -> int:
        return int(XSched.shim_dll.zeCommandQueueSynchronize(ctypes.c_void_p(cmdq), timeout))
