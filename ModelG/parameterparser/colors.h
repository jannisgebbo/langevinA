#ifndef FCN_UTIL_LOG_COLORS_H
#define FCN_UTIL_LOG_COLORS_H
/* File created by Wessel Valkenburg, 2019 */
/* Released under the MIT license, see LICENSE.md. */


/* http://stackoverflow.com/questions/3585846/color-text-in-terminal-aplications-in-unix */
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KTEST1  "\x1B[37m"
#define KTEST2  "\x1B[38m"
#define KRESET "\033[0m"

constexpr const char* ColouredOK() { return "[" KGRN " O K " KRESET "] " ;}
constexpr const char* ColouredERROR() { return "[" KRED "ERROR" KRESET "] " ;}



#endif
