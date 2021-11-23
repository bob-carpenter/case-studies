#ifndef FASTA_PARSER_HPP
#define FASTA_PARSER_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fasta {
namespace internal {

/**
 * Return `true` if the specified line is an identifier beginning with
 * the `>` character, which signals the beginning of a new sequence.
 *
 * @param line line to test
 * @return true if the line contains an identifier
 */
bool start_seq(std::string& line) {
  return line.size() > 0 && line[0] == '>';
}

/**
 * Set the specified line to the last line if the last line is
 * non-empty or try to read a line from the specified input stream,
 * returning true if the variable line was set.
 *
 * @param in input stream from which to read
 * @return true if there was a line
 */
bool getnextline(std::istream& in, const std::string& last,
                 std::string& line) {
  if (!last.empty()) {
    line = last;
    return true;
  }
  while (std::getline(in, line).good())
    if (!line.empty()) {
      return true;
    }
  return false;
}

} // namespace internal
} // namespace fasta


namespace fasta {

/**
 * Return the total number of unique kmers for the specified size.
 *
 * @param K size of kmer (number of bases)
 * @return number of unique kmers
 */
int64_t kmers(int64_t K) {
  return 1UL << (2 * K);  // integer version of 4^K
}

/**
 * Return the identifier (in range 0-3) for the specified base.  The
 * mapping is defined alphabetically, indexing from zero, so that 'A'
 * maps to 0, 'C' maps to 1, 'G' maps to 2, and 'T' maps to 3, with
 * any other input throwing an exception.
 *
 * @param c character representing a base
 * @return numerical identifier for the base
 * @throw std::runtime_error if c is not one of 'A', 'C', 'G', or
 * 'T'.
 */
std::size_t base_id(char c) {
  switch (c) {
    case 'A' : return 0;
    case 'C' : return 1;
    case 'G' : return 2;
    case 'T' : return 3;
    default: throw
        std::runtime_error("argument to base_id must be one of:"
                           "'A', 'C', 'G', 'T'");
  }
}

/**
 * Return a numerical identifier for the specified kmer.  Each kmer is
 * read as a base-4 number with digits defined by the `base_id()`
 * function.  For kmers of size K, this assigns each kmer a
 * unique identifer between 0 and 4^K - 1 inclusive.
 *
 * @param kmer string representing a kmer
 * @return identifier for the kmer.
 * @throw std::runtime_error if there are characters in kmer other
 * than 'A', 'C', 'G', or 'T'.
 */
std::size_t kmer_id(const std::string& kmer) {
  size_t id = 0;
  for (const char& b : kmer)
    id = 4 * id + base_id(b);
  return id;
}

/**
 * Read identifiers and sequences from the FASTA input stream and send
 * them to the specified callback handler, including those with
 * identifiers marked as `PREDICTED` (i.e., not verified) if the flag
 * is set to `true`.
 *
 * The callback handler must be a functor implementing a method
 * `void operator()(const std::string& id, const std::string& seq)`.
 *
 * @tparam F type of callback handler
 * @param[in,out] in input stream
 * @param[in] f callback handler
 * @param include_predicted `true` if sequences marked `PREDICTED` are
 * included
 */
template <typename F>
void parse_stream(std::istream& in, F& f, bool include_predicted = false) {
  std::string line;
  std::string lastline = "";
  while (internal::getnextline(in, lastline, line)) {
    lastline = "";
    if (!internal::start_seq(line)) continue;
    std::string id = line.substr(1);
    if (!include_predicted && id.find("PREDICTED") != std::string::npos)
      continue;
    std::stringstream val;
    while (internal::getnextline(in, lastline, line)) {
      lastline = "";
      if (internal::start_seq(line)) {
        lastline = line;
        break;
      }
      val << line;
    }
    f(id, val.str());
  }
}


/**
 * Convert the specified file name to an input stream and delegate to
 * the stream parser as `parse_file(std::istream&, F&, bool)`, closing
 * the file stream when finished reading.
 *
 * @tparam F type of callback handler
 * @param[in] filename system path to file
 * @param[in] f callback handler
 * @param include_predicted `true` if sequences marked `PREDICTED` are
 * included
 */
template <typename F>
void parse_file(const std::string& filename, F& f,
		bool include_predicted = false) {
  std::ifstream in(filename, std::ifstream::in);
  parse_stream(in, f, include_predicted);
  in.close();
}

} // namespace fasta


#endif
