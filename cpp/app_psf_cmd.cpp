#include <iostream>
#include <string>
#include <regex>
#include <exception>
#include <iomanip>

#include "psf.h"

// from https://stackoverflow.com/a/64886763/2631225
std::vector<std::string> split(const std::string& str, const std::string& regex_str)
{
    std::regex regexz(regex_str);
    std::vector<std::string> list(std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
                                  std::sregex_token_iterator());
    return list;
}

Eigen::MatrixXd parsePointSet(const std::string& input)
{
    auto setPosAsString = split(input, ";");
    std::vector<std::tuple<double,double,double>> points;
    for(const auto& posStr : setPosAsString)
    {
        auto compAsString = split(posStr, ",");
        if( compAsString.size() != 3 )
            throw std::invalid_argument("Three vector components separated by ',' required");

        points.emplace_back( std::stod(compAsString[0]), std::stod(compAsString[1]), std::stod(compAsString[2]) );
    }

    return vectorOfPositions2EigenMatrix(points);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> processInput(const std::string& input)
{
    auto pointSetsAsString = split(input, ":");
    if(pointSetsAsString.size() != 2)
        throw std::invalid_argument("Two point sets separated by ':' required");

    return {parsePointSet(pointSetsAsString[0]), parsePointSet(pointSetsAsString[1])};
}

std::string matAsStr(const std::string& name, const Eigen::MatrixXd& mat)
{
    std::ostringstream oss;
    Eigen::IOFormat matOutputFormat(Eigen::StreamPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
    oss << std::fixed << std::setprecision(5) << name << std::endl << mat.format(matOutputFormat) << std::endl;
    return oss.str();
}

int main(int argc, char** argv)
{
    // argv to string
    std::string input = "";
    for(size_t ac = 1; ac < argc; ac++)
        input.append(argv[ac]);

    auto[setA, setB] = processInput(input);

    Eigen::IOFormat matOutputFormat(Eigen::StreamPrecision, 0, ", ", ";\n", "[", "]", "[", "]");

    std::cout << matAsStr("Point Set A:", setA.topRows(3)) << std::endl;
    std::cout << matAsStr("Point Set B:", setB.topRows(3)) << std::endl;

    auto[ transformation, error ] = pointSetsFitting(setA, setB);

    std::cout << matAsStr("Transformation A -> B:", transformation) << std::endl;
    std::cout << std::fixed << std::setprecision(5) << "Fitting Error: " << error << std::endl;

    return 0;
}