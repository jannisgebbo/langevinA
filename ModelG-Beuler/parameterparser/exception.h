#ifndef FCN_UTIL_EXCEPTION_H
#define FCN_UTIL_EXCEPTION_H
/* File created by Wessel Valkenburg, 2019 */
/* Released under the MIT license, see LICENSE.md. */


#include "colors.h"
#include "stacktrace.h"

namespace FCN {

    
    /** \brief Recursion stopper for helper function for unpacking the variadic arguments of the Exception constructor. */
    inline void ExceptionArgumentUnpacker( std::stringstream& stream) {
        /* stop the recursion. */
    }

    /** \brief Helper function for unpacking the variadic arguments of the Exception constructor. */
    template <typename T, typename... Args>
    inline void ExceptionArgumentUnpacker( std::stringstream& stream, T t, Args... args) {
        stream << t << " ";
        ExceptionArgumentUnpacker(stream, args...);
    }

    
    /** \brief An exception which takes variadic arguments, all recorded into an error message.
     *
     * Unit test: make test-exception
     */
    
    class Exception : public std::exception {
        public:
        /* Put public methods here. These should change very little over time. */
        Exception(const char* itheWhat) : theWhat(itheWhat), strtrace(Stacktrace()) {
            //    std::cerr << "Constructing exception: " << theWhat << "\n" << trace() << "\n";
            theStringWhat = KRED + std::string(itheWhat) + KRESET;
        };
        template <typename... Args>
        Exception(Args... args) :
        theWhat(NULL),
        strtrace(Stacktrace())
        {
            std::stringstream stream;
            ExceptionArgumentUnpacker(stream, args...);
            theStringWhat = KRED + stream.str() + KRESET;
        };
        const char* what() const noexcept
        {
            std::cerr << strtrace << "\n";
            return theStringWhat.length() ? theStringWhat.c_str() : theWhat;
        }
        const std::string& trace() const noexcept { return strtrace; }
            
            
        private:
        /* Put all member variables and private methods here. These may change arbitrarily. */
        const char* theWhat;
        std::string theStringWhat;
        const std::string strtrace;
            
            
            
    };
}

#define MakeException(name) class name : public FCN::Exception {\
public:\
  name(const char* iitheWhat) : FCN::Exception(iitheWhat) { };\
  template <typename... Args> name(Args... args) : FCN::Exception(args...) { };\
}


            
            
#endif
