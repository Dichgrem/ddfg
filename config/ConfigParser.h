#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <unordered_map>

class ConfigParser {
public:
    explicit ConfigParser(const std::string& filepath);
    bool load();
    std::string get(const std::string& key) const;
    void printAll() const;

private:
    std::string _path;
    std::unordered_map<std::string, std::string> _items;
};

#endif // CONFIG_PARSER_H

