#include "kmers/fasta.hpp"
#include <string>

int main() {
  // std::string file = "../data/unpacked/refseq-select-2020-10-22.fasta";
  std::string file = "../data/unpacked/GRCh38_latest_rna.fna";
  size_t K = 10;
  seq_map m(K, file);

  std::cout << std::endl;
  std::cout << "K = " << K
	    << std::endl;
  std::cout << m.size() << " identifiers"
            << std::endl;
  std::cout << m.total_bases() << " bases"
            << std::endl;
  std::cout << "Calculating k-mer frequency histogram."
	    << std::endl;

  counter<size_t> c = m.kmer_frequency_counts();
  std::cout << std::endl
	    << "# K-MER FREQUENCY HISTOGRAM (entries = " << c.counts_.size() << ")"
	    << std::endl;
  std::cout << "freq,count"
	    << std::endl;
  long total_kmer_count = 0;
  for (const auto& freq_count : c.counts_) {
    total_kmer_count += freq_count.second;
    std::cout << "k-mer frequency = " << freq_count.first
              << ";  count = " << freq_count.second
	      << std::endl;
  }
  int missing_kmer_count = static_cast<int>(std::pow(4, K) - total_kmer_count);
  std::cout << "k-mer frequency = " << 0
	    << ";  count = " << missing_kmer_count
	    << std::endl;
  std::cout << "4^K = " << std::pow(4, K) << ";  total count=" << total_kmer_count
	    << std::endl;
  std::cout << "=========================================="
	    << std::endl;
  std::cout << total_kmer_count << " distinct k-mers in the transcriptome"
	    << std::endl;

  // Eigen::VectorXd a(3);  a << 1, 2, 3;
  // Eigen::VectorXd sm_a = softmax(a);

  // std::cout << "K-mer id  = " << kmer_id("") << std::endl;
  // std::cout << "K-mer id A = " << kmer_id("AA") << std::endl;
  // std::cout << "K-mer id AA = " << kmer_id("AA") << std::endl;
  // std::cout << "K-mer id AC = " << kmer_id("AC") << std::endl;
  // std::cout << "K-mer id CA = " << kmer_id("CA") << std::endl;

  // std::cout << "Calculating k-mer by gene matrix" << std::endl;
  // seq_map::mat_t x = m.kmer_gene_matrix();


  // std::cout << "gradient at 0" << std::endl;
  // Eigen::VectorXd y = Eigen::VectorXd::Zero(m.size());
  // log_posterior_gradient lp_g(1, y, x);
  // Eigen::VectorXd grad;
  // Eigen::VectorXd alpha = Eigen::VectorXd::Zero(static_cast<size_t>(m.size()));
  // std::cout << std::endl << "alpha.size() = " << alpha.size() << std::endl;
  // double lp = lp_g(alpha, grad);
  // std::cout << std::endl << "lp = " << lp << std::endl;
}
