#ifndef FCN_UTIL_STRINGTRIMMER_H
#define FCN_UTIL_STRINGTRIMMER_H
/* File created by Wessel Valkenburg, 2019 */
/* Released under the MIT license, see LICENSE.md. */


/* from http://stackoverflow.com/a/217605 */
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>


namespace FCN {
    
    /** \brief A class which trims strings on all ends.
     * In place: [lr]trim(std::string).
     * Copy: std::string [lr]trimmed(std::string)
     *
     * Unit test: make test-stringtrimmer
     */
    
    class StringTrimmer {
        public:
        /* Put public methods here. These should change very little over time. */
        static inline void ltrim(std::string &s) {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                            std::not1(std::ptr_fun<int, int>(std::isspace))));
        }
        
        // trim from end (in place)
        static inline void rtrim(std::string &s) {
            s.erase(std::find_if(s.rbegin(), s.rend(),
                                 std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        }
        
        // trim from both ends (in place)
        static inline void trim(std::string &s) {
            ltrim(s);
            rtrim(s);
        }
        
        // trim from start (copying)
        static inline std::string ltrimmed(std::string s) {
            ltrim(s);
            return s;
        }
        
        // trim from end (copying)
        static inline std::string rtrimmed(std::string s) {
            rtrim(s);
            return s;
        }
        
        // trim from both ends (copying)
        static inline std::string trimmed(std::string s) {
            trim(s);
            return s;
        }
        
    };
}



#endif
