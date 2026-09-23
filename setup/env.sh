# Source this to make ORCA / OpenMPI / Multiwfn available on this machine.
# usage:  source setup/env.sh
export ORCA_DIR=$HOME/software/orca_6_1_0_linux_x86-64_shared_openmpi418
export OPENMPI_DIR=$HOME/software/openmpi-4.1.8-install
export MULTIWFN_DIR=$HOME/software/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn_3.8_bin_Linux_noGUI

# put ORCA first so /usr/bin/orca (GNOME screen reader) never wins
export PATH=$ORCA_DIR:$OPENMPI_DIR/bin:$MULTIWFN_DIR:$PATH
export LD_LIBRARY_PATH=$ORCA_DIR:$OPENMPI_DIR/lib:${LD_LIBRARY_PATH:-}

# Multiwfn needs its dir on Multiwfnpath (settings.ini uses it)
export Multiwfnpath=$MULTIWFN_DIR

# OMP for Multiwfn (Section 2.1.2 of the manual)
export OMP_STACKSIZE=200M
ulimit -s unlimited 2>/dev/null || true
