#pragma once

#include <cstdint>

#define PREEMPT_MODE_STOP_SUBMISSION    1
#define PREEMPT_MODE_DEACTIVATE         2
#define PREEMPT_MODE_INTERRUPT          3

#ifdef __cplusplus
#define ZE_CTRL_FUNC extern "C" __attribute__((visibility("default")))
#else
#define ZE_CTRL_FUNC __attribute__((visibility("default")))
#endif

typedef int32_t Prio;
typedef int32_t Bwidth;
typedef int64_t DeadlineUs;
typedef int64_t LaxityUs;
typedef struct _ze_command_queue_handle_t *ze_command_queue_handle_t;

ZE_CTRL_FUNC uint64_t ZeXQueueCreate(ze_command_queue_handle_t cmdq,
                                     int preempt_mode,
                                     int64_t queue_length,
                                     int64_t sync_interval);
ZE_CTRL_FUNC void ZeXQueueDestroy(uint64_t handle);

ZE_CTRL_FUNC void ZeXQueueSuspend(ze_command_queue_handle_t cmdq,
                                  bool sync_hal_queue = false);
ZE_CTRL_FUNC void ZeXQueueResume(ze_command_queue_handle_t cmdq,
                                 bool drop_commands = false);

ZE_CTRL_FUNC void ZeSetOverrideHalQueue(int32_t queue_idx);
ZE_CTRL_FUNC void ZeUnsetOverrideHalQueue();
ZE_CTRL_FUNC ze_command_queue_handle_t ZeGetOverrideHalQueue(int32_t queue_idx);

ZE_CTRL_FUNC void ZeXQueueSetPriority(ze_command_queue_handle_t cmdq,
                                      Prio prio);
ZE_CTRL_FUNC void ZeXQueueSetBandwidth(ze_command_queue_handle_t cmdq,
                                       Bwidth bdw);
ZE_CTRL_FUNC void ZeXQueueSetDeadline(ze_command_queue_handle_t cmdq,
                                      DeadlineUs deadline);
ZE_CTRL_FUNC void ZeXQueueSetLaxity(ze_command_queue_handle_t cmdq,
                                    LaxityUs laxity,
                                    Prio lax_prio,
                                    Prio critical_prio);
