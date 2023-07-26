#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// This function takes a string representing a single line of a CSV file
// and parses it into a vector of strings representing the individual fields
// in the line.
std::vector<std::string> parse_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream line_stream(line);
  std::string field;
  while (getline(line_stream, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

int main() {
  // Open the CSV file for reading
  std::ifstream csv_file("file.csv");

  // Read the file line by line
  std::string line;
  while (getline(csv_file, line)) {
    // Parse each line and add the resulting fields to an array
    std::vector<std::string> fields = parse_csv_line(line);
    std::cout << "Field 1: " << fields[0] << std::endl;
    std::cout << "Field 2: " << fields[1] << std::endl;
    std::cout << "Field 3: " << fields[2] << std::endl;
    std::cout << "Field 4: " << fields[3] << std::endl;
    std::cout << "Field 5: " << fields[4] << std::endl;
  }

  return 0;
}
