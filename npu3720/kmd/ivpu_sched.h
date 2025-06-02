#ifndef __IVPU_SCHED_H__
#define __IVPU_SCHED_H__

#include <linux/kref.h>
#include <linux/idr.h>

#include "ivpu_accel_override.h" // overrides include/uapi/drm/ivpu_accel.h
#include "ivpu_gem.h"

int ivpu_schedule_ioctl(struct drm_device *dev, void *data, struct drm_file *file);

#endif /* __IVPU_SCHED_H__ */
