#ifndef FCN_UTIL_DEBUG_STACKTRACE_H
#define FCN_UTIL_DEBUG_STACKTRACE_H
/* File created by Wessel Valkenburg, 2019 */
/* Released under the MIT license, see LICENSE.md. */


#include "stacktraceplainptrs.h"
#include "stacktraceptrtofileaddr.h"
#include "cdemangle.h"

namespace FCN {
    
    /** \brief A class which holds the stacktrace to the point where it was
     * instantiated.
     *
     *
     * Unit test: make test-stacktrace
     */
    
    class Stacktrace {
        public:
        /* Put public methods here. These should change very little over time. */
        Stacktrace(ptrdiff_t skipBegin = 0, ptrdiff_t maxSize = 128) {
            CollectStack(skipBegin, maxSize);
        }
        
        operator std::string() { return value; }
        
        friend std::ostream& operator<< (std::ostream& ostream, const Stacktrace& sttrc) {
            ostream << sttrc.value;
            return ostream;
        }
        
        private:
        /* Put all member variables and private methods here. These may change arbitrarily. */
        std::string value;
        void CollectStack(ptrdiff_t skipBegin, ptrdiff_t maxSize) {
            std::stringstream stream;
            stream << "FCN's own stack trace:\n";
            
            
#ifdef FCNPOSIXDETECTED
            // storage array for stack trace address data
            std::array<void*, 128> addrlist;
            // retrieve current stack addresses
            int addrlen;
            StacktracePlainptrs(&addrlist, &addrlen);
            
            if (addrlen == 0) {

                stream << "  <empty, possibly corrupt>\n";

            } else {
                
                // resolve addresses into strings containing "filename(function+address)",
                // this array must be free()-ed
                char** symbollist = backtrace_symbols(addrlist.data(), addrlen);
                
                ptrdiff_t offset = StacktracePtrToFileAddress();
                
                // iterate over the returned symbol lines. skip the first, it is the
                // address of this function.
                for (ptrdiff_t i = 1 + skipBegin; i < std::min((ptrdiff_t) addrlen, 1 + skipBegin + maxSize); i++)
                {
                    stream << "plain: " << addrlist[i] << " offset: " << (void*) ((char*)addrlist[i] - offset) << "\n";
                    CDemangle(stream, symbollist[i]);
                }
                
                free(symbollist);
            }
#endif // posix
            
            stream << "Base address: " << StacktracePtrToFileAddress() << "\n";
            value = stream.str();
        }
        
    };
}



#endif
