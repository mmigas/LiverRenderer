#include <mitsuba/core/thread.h>
#include <mitsuba/core/logger.h>
#include "python.h"

MTS_PY_DECLARE(filesystem);
MTS_PY_DECLARE(pcg32);
MTS_PY_DECLARE(atomic);
MTS_PY_DECLARE(util);
MTS_PY_DECLARE(math);
MTS_PY_DECLARE(xml);
MTS_PY_DECLARE(vector);
MTS_PY_DECLARE(Object);
MTS_PY_DECLARE(Thread);
MTS_PY_DECLARE(Logger);
MTS_PY_DECLARE(Appender);
MTS_PY_DECLARE(Formatter);
MTS_PY_DECLARE(Properties);
MTS_PY_DECLARE(ArgParser);
MTS_PY_DECLARE(FileResolver);
MTS_PY_DECLARE(Stream);
MTS_PY_DECLARE(DummyStream);

PYBIND11_PLUGIN(mitsuba) {
    Class::staticInitialization();
    Thread::staticInitialization();
    Logger::staticInitialization();

    py::module m("mitsuba", "Mitsuba Python extension library");

    MTS_PY_IMPORT(filesystem);
    MTS_PY_IMPORT(pcg32);
    MTS_PY_IMPORT(atomic);
    MTS_PY_IMPORT(util);
    MTS_PY_IMPORT(math);
    MTS_PY_IMPORT(xml);
    MTS_PY_IMPORT(vector);
    MTS_PY_IMPORT(Object);
    MTS_PY_IMPORT(Thread);
    MTS_PY_IMPORT(Logger);
    MTS_PY_IMPORT(Appender);
    MTS_PY_IMPORT(Formatter);
    MTS_PY_IMPORT(Properties);
    MTS_PY_IMPORT(ArgParser);
    MTS_PY_IMPORT(FileResolver);
    MTS_PY_IMPORT(Stream);
    MTS_PY_IMPORT(DummyStream);

    atexit([](){
        Logger::staticShutdown();
        Thread::staticShutdown();
        Class::staticShutdown();
    });

    return m.ptr();
}
