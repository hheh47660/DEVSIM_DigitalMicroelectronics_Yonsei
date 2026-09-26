# HOWTO
## Update version number

To update the version number update the lines in these files:

* ``CMakeLists.txt``
```
ADD_DEFINITIONS(-DDEVSIM_VERSION_STRING=\"2.10.0\")
```
* ``dist/bdist_wheel/setup.cfg``
```
version = 2.9.2
```

## Update minimum python version
* ``dist/bdist_wheel/setup.cfg``
```
py-limited-api = cp39
```
* ``src/pythonapi/CMakeLists.txt``
```
target_compile_definitions(pythonapi_interpreter_py3 PRIVATE -DDEVSIM_MODULE_NAME=devsim_py3 -DPy_LIMITED_API=0x03090000)
```
* Additional Python Notes
  * [how-to-configure-setuptools-with-setup-cfg-to-include-platform-name-python-tag](https://stackoverflow.com/questions/72090919/how-to-configure-setuptools-with-setup-cfg-to-include-platform-name-python-tag)
  * [C API Stability](https://docs.python.org/3/c-api/stable.html)
  * It looks like ``setup.cfg`` is going away, but it is not clear what to do for the replacement ``pyproject.toml``.
