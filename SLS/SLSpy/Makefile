PY=
ifneq (,$(shell which python))
  PY=python
endif
ifneq (,$(shell which python3))
  PY=python3
endif

PYTHON_VERSION=$(shell $(PY) -c "import sys;t='{v[0]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")

ifneq (,$(shell which dnf))
  # for Fedora
  PM=dnf
  LIBS=blas blas-devel lapack lapack-devel
  ifeq ($(PYTHON_VERSION),3)
    LIBS+= python3-devel python3-tkinter
  else
    LIBS+= python-devel python-tkinter
  endif
else
ifneq (,$(shell which yum))
  # for CentOS
  PM=yum
  LIBS=blas blas-devel lapack lapack-devel
  ifeq ($(PYTHON_VERSION),3)
    LIBS+= python3-devel python3-tkinter
  else
    LIBS+= python-devel python-tkinter
  endif
else
ifneq (,$(shell which apt))
  # for Ubuntu, Debian
  PM=apt
  LIBS=libblas-dev liblapack-dev
  ifeq ($(PYTHON_VERSION),3)
    LIBS+= libpython3-dev python3-tk
  else
    LIBS+= libpython-dev python-tk
  endif
else
ifneq (,$(shell which pkg))
  # FreeBSD, not yet supported
  PM=pkg
  $(info FreeBSD is not yet supported)
else
  PM=
endif
endif
endif
endif
  
# install correct pip
ifeq ($(PYTHON_VERSION),3)
  PIP=pip3
  ifeq (, $(shell which pip3))
    LIBS+= python3-pip
  endif
else
  PIP=pip
  ifeq (, $(shell which pip))
    LIBS+= python-pip
  endif
endif

setup:
ifneq ($(PM),)
	$(PM) install $(LIBS)
endif
	# install numpy first to avoid weird scs dependency issues
	$(PIP) install numpy
	$(PIP) install --no-cache-dir -I scs
	$(PIP) install cvxpy matplotlib

install:
	$(PIP) install -e .

uninstall:
	$(PIP) uninstall slspy
