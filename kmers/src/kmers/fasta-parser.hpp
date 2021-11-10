#ifndef FASTA_PARSER_HPP
#define FASTA_PARSER_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fasta {
namespace internal {

bool start_seq(std::string& line) {
  return line.size() > 0 && line[0] == '>';
}

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
 * Return the total number of unique k-mers for the specified number
 * of bases.
 *
 * @param K number of bases
 * @return total number of unique k-mers
 */
int64_t kmers(int64_t K) {
  return 1UL << (2 * K);
}

/**
 * Return the identifier in the range 0-3 for the specified base.
 * The mapping is defined alphabetically, indexing from zero, so that
 * 'A' maps to 0, 'C' maps to 1, 'G' maps to 2, and 'T' maps to 3,
 * with any other input throwing an exception.
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
 * Return a numerical identifier for the specified kmer.  Each k-mer
 * is read as a base-4 number given the base_id() for each k-mer.  For
 * a fixed K, this assigns each K-mer gets a unique ID between 0 and
 * 4^K - 1 based on a lexicographic start, indexing from 0.  The
 * mapping is A = 0, C = 1, G = 2, and T = 3.
 *
 * @param kmer string representing a k-mer
 * @return identifier for the k-mer.
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
 * Parse the data in fasta format from the specified inputs stream,
 * sending reads to the specified callback handler.  The callback
 * handler must be a functor implementing `void operator()(const
 * std::string& id, const std::string& seq)`.
 *
 * @tparam F type of callback handler
 * @param[in,out] in input stream
 * @param[in] f callback handler
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
 * Parse the data in fasta form from the specified file, sending reads
 * to the specified callback handler.The callback
 * handler must be a functor implementing `void operator()(const
 * std::string& id, const std::string& seq)`.
 *
 * @tparam F type of callback handler
 * @param[in] filename system path to file
 * @param[in] f callback handler
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
