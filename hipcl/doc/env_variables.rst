ENV variables controlling behaviour
---------------------------------------

The behavior of HIPLZ can be controlled with multiple environment variables
listed below. The variables are helpful both when using and when developing
pocl.

- **HIPLZ_LOGLEVEL**
  String value. Changes verbosity of log messages coming from HIPLZ.
  Possible values are: debug,info,warn,err,crit,off
  Defaults to "err". HIPLZ will log messages of this priority and higher.

- **HIPLZ_PLATFORM**
  Numeric value. If there are multiple OpenCL platforms on the system, setting this to a number (0..platforms-1)
  will limit HipCL to that single platform. By default HipCL can access all OpenCL platforms.

- **HIPLZ_DEVICE**
  Numeric value. If there are multiple OpenCL devices in the selected platform, setting this to a number (0..N-1)
  will limit HipCL to a single device. If HIPLZ_PLATFORM is not set but HIPLZ_DEVICE is,
  HIPLZ_PLATFORM defaults to 0.

- **HIPLZ_DEVICE_TYPE**
  String value. Limits OpenCL device visibility to HIPLZ based on device type.
  Possible values are: all, cpu, gpu, default, accel

