#!/bin/bash
mv libpil.py libpil.pyx
python setup.py build_ext --inplace
mv libpil.pyx libpil.py
