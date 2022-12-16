STEP1="step1/main.cpp step1/Makefile step1/nbody.cpp step1/nbody.h"
STEP1_MATRIX="step1_matrix/main.cpp step1_matrix/Makefile step1_matrix/nbody.cpp step1_matrix/nbody.h"
STEP2="step2/main.cpp step2/Makefile step2/nbody.cpp step2/nbody.h"
STEP3="step3/main.cpp step3/Makefile step3/nbody.cpp step3/nbody.h"
STEP3_SEQ="step3_reduction_seq/main.cpp step3_reduction_seq/Makefile step3_reduction_seq/nbody.cpp step3_reduction_seq/nbody.h"
STEP4="step4/main.cpp step4/Makefile step4/nbody.cpp step4/nbody.h"
STEP4_CPU="step4_cpu/main.cpp step4_cpu/Makefile step4_cpu/nbody.cpp step4_cpu/nbody.h"

SCRIPTS="run.sh"
OUT_FILE="nbody.txt"
zip xpolok03.zip ${STEP1} ${STEP1_MATRIX} ${STEP2} ${STEP3} ${STEP3_SEQ} ${STEP4} $STEP4_CPU ${SCRIPTS} ${OUT_FILE}