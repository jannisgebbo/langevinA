#ifndef FCN_PARAMETERS_FILEREADER_H
#define FCN_PARAMETERS_FILEREADER_H
/* File created by Adrien Florio, 2019 */
/* Released under the MIT license, see LICENSE.md. */

#include <fstream>
#include "exception.h"
#include "stringtrimmer.h"

namespace FCN {

    /** \brief A class which
     * \todo Write this.
     *
     *
     * Unit test: make test-filereader
     **/

    MakeException(FileReaderProblemInputFile);

    class FileReader {
    public:
        /* Put public methods here. These should change very little over time. */
        FileReader() {

        }
        const std::string& operator[](int i) const {
          return vec[i];
        }
        size_t size() const
        {
          return vec.size();
        }
        void operator()(const std::string& str, char comment = '#')
        {
            std::ifstream t;
            t.open(str);
            std::string tmp;
            if(t.good()){
              while(getline(t,tmp)){
                StringTrimmer::trim(tmp);
                if(tmp != "" && tmp[0] != comment)
                  vec.push_back(tmp);
              }
            }
            else throw(FileReaderProblemInputFile("There was a problem opening the input file. Abort."));
        }

    private:
        /* Put all member variables and private methods here. These may change arbitrarily. */

        std::vector<std::string> vec;
    };
}



#endif
