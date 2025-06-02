#include "ivpu_accel_override.h" // overrides include/uapi/drm/ivpu_accel.h

#include <drm/drm_file.h>

#include <linux/bitfield.h>
#include <linux/highmem.h>
#include <linux/pci.h>
#include <linux/module.h>
#include <uapi/drm/ivpu_accel.h>

#include "ivpu_drv.h"
#include "ivpu_fw.h"
#include "ivpu_hw.h"
#include "ivpu_ipc.h"
#include "ivpu_job.h"
#include "ivpu_jsm_msg.h"
#include "ivpu_pm.h"
#include "ivpu_sched.h"
#include "vpu_boot_api.h"

static inline u8 ivpu_job_to_hws_priority(struct ivpu_file_priv *file_priv, u8 priority)
{
	if (priority == DRM_IVPU_JOB_PRIORITY_DEFAULT)
		return DRM_IVPU_JOB_PRIORITY_NORMAL;

	return priority - 1;
}

static int ivpu_sched_cmdq(struct ivpu_file_priv *file_priv, u32 engine, u8 priority, u32 operation)
{
	struct ivpu_device *vdev = file_priv->vdev;
	struct ivpu_cmdq *cmdq;
	int cmdq_idx;
	int ret;

	mutex_lock(&file_priv->lock);
	cmdq_idx = IVPU_CMDQ_INDEX(engine, priority);
	cmdq = file_priv->cmdq[cmdq_idx];
	mutex_unlock(&file_priv->lock);

	if (!cmdq) {
		ivpu_warn_ratelimited(vdev, "Failed to get cmdq, ctx %d engine %d prio %d\n",
							  file_priv->ctx.id, engine, priority);
		ret = -EINVAL;
		return ret;
	}

	switch (operation) {
	case DRM_IVPU_SCHED_OP_CMDQ_SUSPEND:
		// the doorbell id is the cmdq id
		ret = ivpu_jsm_hws_suspend_cmdq(vdev, file_priv->ctx.id, cmdq->db_id);
		break;
	case DRM_IVPU_SCHED_OP_CMDQ_RESUME:
		ret = ivpu_jsm_hws_resume_cmdq(vdev, file_priv->ctx.id, cmdq->db_id);
		break;
	default:
		ivpu_warn_ratelimited(vdev, "Invalid cmdq schedule operation: %d, ctx %d engine %d prio %d\n",
							  operation, file_priv->ctx.id, engine, priority);
		ret = -EINVAL;
		break;
	}

	return ret;
}

static int ivpu_sched_engine(struct ivpu_file_priv *file_priv, u32 engine, u32 operation, u32 request_id)
{
	struct ivpu_device *vdev = file_priv->vdev;
	int ret;

	switch (operation) {
	case DRM_IVPU_SCHED_OP_ENGINE_PREEMPT:
		ret = ivpu_jsm_preempt_engine(vdev, engine, request_id);
		break;
	case DRM_IVPU_SCHED_OP_ENGINE_RESET:
		ret = ivpu_jsm_reset_engine(vdev, engine);
		break;
	case DRM_IVPU_SCHED_OP_ENGINE_RESUME:
		ret = ivpu_jsm_hws_resume_engine(vdev, engine);
		break;
	default:
		ivpu_warn_ratelimited(vdev, "Invalid engine schedule operation: %d, ctx %d engine %d\n",
							  operation, file_priv->ctx.id, engine);
		ret = -EINVAL;
		break;
	}

	return ret;
}

int ivpu_schedule_ioctl(struct drm_device *dev, void *data, struct drm_file *file)
{
	struct ivpu_file_priv *file_priv = file->driver_priv;
	struct ivpu_device *vdev = file_priv->vdev;
	struct drm_ivpu_schedule *params = data;
	u8 priority;
	int ret;

	// drm_info(dev, "ivpu_schedule_ioctl called, priority: %u, operation: %u\n",
	// 		 params->priority, params->operation);

	switch (params->operation) {
	case DRM_IVPU_SCHED_OP_CMDQ_RESUME:
	case DRM_IVPU_SCHED_OP_CMDQ_SUSPEND:
		priority = ivpu_job_to_hws_priority(file_priv, params->priority);
		ret = ivpu_sched_cmdq(file_priv, params->engine, priority, params->operation);
		break;
	case DRM_IVPU_SCHED_OP_ENGINE_PREEMPT:
	case DRM_IVPU_SCHED_OP_ENGINE_RESET:
	case DRM_IVPU_SCHED_OP_ENGINE_RESUME:
		ret = ivpu_sched_engine(file_priv, params->engine, params->operation, params->request_id);
		break;
	default:
		ivpu_warn_ratelimited(vdev, "Invalid schedule operation: %d, ctx %d\n",
							  params->operation, file_priv->ctx.id);
		ret = -EINVAL;
		break;
	}

	return ret;
}
