#include <iostream>
#include <string>
#include <regex>

#include "psf.h"

// from https://stackoverflow.com/a/64886763/2631225
std::vector<std::string> split(const std::string& str, const std::string& regex_str)
{
    std::regex regexz(regex_str);
    std::vector<std::string> list(std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
                                  std::sregex_token_iterator());
    return list;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> processInput(const std::string& input)
{
    auto pointSetsAsString = split(input, ":");

}


int main(int argc, char** argv)
{
    // argv to string
    std::string input = "";
    for(size_t ac = 1; ac < argc; ac++)
        input.append(argv[ac]);

    auto[setA, setB] = processInput(input);



    std::cout << input << std::endl;

    return 0;
}