#ifndef FCN_PARAMETERS_STRINGCONVERTER_H
#define FCN_PARAMETERS_STRINGCONVERTER_H
/* File created by Adrien Florio, 2019 */
/* Released under the MIT license, see LICENSE.md. */

#include <vector>
#include <iomanip>

namespace FCN {

    /** \brief A class which wraps a single function, splitting it in lines and passing each
     *  line to ParameterGetter, returning the result in your provided MultipleParameterGetter<T>.
     *
     **/
    template<class T>
    class StringConverter {
    public:
        /* Put public methods here. These should change very little over time. */
        StringConverter() {

        }
         void operator()(const std::string& str, std::vector<T>& arr ,const std::string& name)
        {
            arr.clear();
            T tmp;
            std::istringstream iss(str);

            while(iss>>std::boolalpha>>std::skipws>>tmp){
                arr.push_back(tmp);
            }
        }
    };
}



#endif
