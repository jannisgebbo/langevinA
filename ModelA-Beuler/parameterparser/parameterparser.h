#ifndef FCN_PARAMETERS_PARAMETERS_H
#define FCN_PARAMETERS_PARAMETERS_H
/* File created by Adrien Florio, 2019 */
/* Released under the MIT license, see LICENSE.md. */

#include "pairmaker.h"
#include <unordered_map>
#include "stringconverter.h"
#include "filereader.h"
#include <random>
#include "exception.h"
#include <iomanip>
#include <iostream>

namespace FCN {

    /** \brief The outside-world interface for parameter parsing. Pass your argc/argv, receive
     *   an object that parsed everything that you put on the command line interface,
     *   where 'everything' means that you pass arguments in the form:
     *
     *   a=1 b=2 c=3
     *
     *   The special keyword 'input' takes a filename as value, pointing to a file
     *   in which you put all the parameters at once.
     *
     *
     * Unit test: make test-parameters
     **/
    MakeException(ParameterParserMissingMandatory);
    MakeException(ParameterParserMismatchSizes);
    MakeException(ParameterParserMismatchDefaultSizes);
    MakeException(ParameterParserDoesNotExist);
    MakeException(ParameterParserIsEmpty);

    class ParameterParser {
    public:

        /* Put public methods here. These should change very little over time. */
        ParameterParser(int argc, char* argv[]) :
        params(),
        pairMaker()
        {
          for (int i = 1; i < argc; ++i){
            insert(argv[i]);
          }
          if(params.count("input")>0){
            FileReader fr;
            fr(params["input"]);
            for (size_t i = 0; i < fr.size(); ++i){
              insert(fr[i]);
            }
          }
        }
        void addFromVector(const  std::vector<std::string>& vec)
        {
            for(auto x : vec) {
                insert(x);
            }
        }

        template<class S>
        S get(const std::string& name){
          return this->get<S, 1>(name, true, {S()}, false)[0];
        }
        template<class S>
        S get(const std::string& name, S def){
          return this->get<S, 1>(name, false, {def}, false)[0];
        }
        template<class S, int N>
        std::vector<S>  get(const std::string& name){
          return this->get<S, N>(name, true, {}, false);
        }
        template<class S, int N>
        std::vector<S>  get(const std::string& name, const std::vector<S>& def){
          return this->get<S, N>(name, false, def, false);
        }

        template<class S>
        S getOverride(const std::string& name){
            return this->get<S, 1>(name, true, {S()}, true)[0];
        }
        template<class S>
        S getOverride(const std::string& name, S def){
            return this->get<S, 1>(name, false, {def}, true)[0];
        }
        template<class S, int N>
        std::vector<S>  getOverride(const std::string& name){
            return this->get<S, N>(name, true, {}, true);
        }
        template<class S, int N>
        std::vector<S>  getOverride(const std::string& name, const std::vector<S>& def){
            return this->get<S, N>(name, false, def, true);
        }

        ptrdiff_t getSeed(const std::string& name)
        {
          ptrdiff_t tmp = this->get<ptrdiff_t>(name,0);
          if(tmp==0)
          {
            std::random_device r;
            tmp=r();//tmp=std::to_string(r());
          }
          return tmp;
        }

        const auto& getParams() const
        {
          return params;
        }

        void insert(const std::string& str){ //Insert value of the current PairMaker in the unordered_map. If already exist, skip it.
            pairMaker(str);

            if(params.count(pairMaker.getKey()) == 0){

              params[pairMaker.getKey()] = pairMaker.getValue();

              if(params[pairMaker.getKey()] == "") throw(ParameterParserIsEmpty("Parameter "+pairMaker.getKey()+" is empty. Abort. If it is an optional parameter, remove it or comment it out."));
            }
            else {
                //say << "Parameter " + pairMaker.getKey() + " already assigned, skipped. The last value of this parameter which is skipped will be cached to have the potential of overriding it.";
                paramsOverride[pairMaker.getKey()] = pairMaker.getValue(); //Store last of the extra value here.
            }

        }

    private:
        /* Put all member variables and private methods here. These may change arbitrarily. */
        std::unordered_map<std::string, std::string> params;
        std::unordered_map<std::string, std::string> paramsOverride;
        PairMaker pairMaker;


        friend std::ostream& operator<<(std::ostream& os, ParameterParser& par){
          for (auto i: par.params)
            os << i.first << " " << i.second << std::endl;
          return os;
        }

        template<class S, int N>
        std::vector<S>  get(const std::string& name, const bool& mandatory, const std::vector<S>& def, bool override){
		std::vector<S>  ret;
          StringConverter<S> conv;
          if((!override || paramsOverride.count(name) == 0) && params.count(name)>0){
                 conv(params[name], ret, name);
                 if(N>0 && ret.size() != N) throw(ParameterParserMismatchSizes("Parameter "+name+" was provided with an unexpected number of arguments. Abort."));
          }
          else if(override &&  paramsOverride.count(name)>0)
          {
              conv(paramsOverride[name], ret, name);
              if(N>0 && ret.size() != N) throw(ParameterParserMismatchSizes("Parameter "+name+" was provided with an unexpected number of arguments. Abort."));
          }
          else if(mandatory) throw(ParameterParserMissingMandatory("Mandatory parameter "+name+" was not specified. Abort."));
          else if(N>0 && def.size() != N) throw(ParameterParserMismatchDefaultSizes("Optional parameter "+name+" was not provided with the correct number of default values. Abort."));
          else{
            std::ostringstream oss;
            oss << name << "=";
            for (int i = 0; i < N; ++i){
              ret.push_back(def[i]);
              oss << std::boolalpha << def[i] << " ";
            }
	    //std::cout << oss.str() << std::endl;
            insert(oss.str()); //useful to insert the optionnal params so we can retrieve them later, for saving purposes for example.
          }
	  //std::cout << "Parameter initialisation: " << ret ;
          return ret;
        }
    };
}


#endif
