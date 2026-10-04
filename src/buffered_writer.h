#pragma once
#include <charconv>
#include <fstream>
#include <string>
#include <system_error>

// Buffers text for an output file, formatting numbers with std::to_chars, which is much faster
// than formatting them through a stream. Doubles and floats are written as an ostream writes
// them by default (as if by printf("%.6g")), so the output is the same.
class OutputBuffer {
public:
    explicit OutputBuffer(std::ofstream& setFile) : file{setFile} {
        buffer.reserve(flushSize + 256);
    }
    ~OutputBuffer() {flush();}

    void add(double value) {addNumber(value, std::chars_format::general, 6);}
    void add(float value) {addNumber(value, std::chars_format::general, 6);}
    void add(int value) {addNumber(value);}
    void add(char character) {buffer.push_back(character);}

    void flush() {
        file.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        buffer.clear();
    }

private:
    static constexpr std::size_t flushSize{1 << 20};
    std::ofstream& file;
    std::string buffer;

    template <typename T, typename... FormatArguments>
    void addNumber(T value, FormatArguments... formatArguments) {
        char characters[32];
        const std::to_chars_result result{std::to_chars(
            characters, characters + sizeof(characters), value, formatArguments...
        )};
        buffer.append(characters, result.ptr);
        if (buffer.size() >= flushSize) {flush();}
    }
};
