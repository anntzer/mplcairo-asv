asv benchmark suite for mplcairo
================================

Run e.g. with ::

   PIP_FIND_LINKS=/path/containing/wheels PIP_NO_INDEX=true asv run

where ``/path/containing/wheels`` contains wheels for all mplcairo,
setuptools_scm and pytest direct and indirect dependencies (as generated by
``pip wheel mplcairo setuptools_scm pytest``).  In particular, the Matplotlib
wheel should be built *without* the ``local_freetype`` option, which older
versions of mplcairo do not support (this is the reason to rely on a local
wheel directory).
