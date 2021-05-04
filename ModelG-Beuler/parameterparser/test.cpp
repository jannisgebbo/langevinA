#include <iostream>
#include "parameterparser.h"


int main(int argc, char* argv[]){

	FCN::ParameterParser par(argc, argv);
	
	int test1 = par.get<int>("my_int"); //Mandatory parameter, program crash if not given in command line or in the file.
	double test2 = par.get<double>("my_double", 12.3); //Optional parameter, default value is 12.3.
	bool test3 = par.get<bool>("my_bool");  // Can give bool as "true/false" or "0/1"

	std::vector<double> test4 = par.get<double, 5>("my_vector"); //Can also pass multiple parameters on a single line, separated by blanks (here 5 parameters).

	std::cout << test1 << std::endl;
	std::cout << test2 << std::endl;
	std::cout << test3 << std::endl;

	for(auto x : test4) std::cout<< x << " ";
	std::cout<<std::endl;

	return 0;

}
