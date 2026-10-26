#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nanomt {
struct Config { uint32_t vocab=0,d_model=0,d_ff=0,n_heads=0,n_encoder=0,n_decoder=0,max_src=0,max_tgt=0,pad=0,bos=1,eos=2,unk=3; };
struct Piece { const uint8_t* data=nullptr; uint16_t len=0; };
struct Q8Tensor { uint32_t rows=0, cols=0; const float* scales=nullptr; const int8_t* q=nullptr; };
struct F32Vec { uint32_t n=0; const float* data=nullptr; };
struct AttentionWeights { Q8Tensor q,k,v,o; };
struct EncoderLayer { F32Vec n1; AttentionWeights attn; F32Vec n2; Q8Tensor w1,w2; };
struct DecoderLayer { F32Vec n1; AttentionWeights self_attn; F32Vec n2; AttentionWeights cross_attn; F32Vec n3; Q8Tensor w1,w2; };

class Model {
public:
    bool load(const void* data,size_t size,std::string* error=nullptr);
    const Config& config() const { return cfg_; }
    const std::vector<Piece>& pieces() const { return pieces_; }
    const Q8Tensor& token_embedding() const { return tok_; }
    const Q8Tensor& src_pos() const { return src_pos_; }
    const Q8Tensor& tgt_pos() const { return tgt_pos_; }
    const std::vector<EncoderLayer>& enc() const { return enc_; }
    const std::vector<DecoderLayer>& dec() const { return dec_; }
    const F32Vec& final_norm() const { return final_; }
private:
    Config cfg_{}; std::vector<Piece> pieces_; Q8Tensor tok_{},src_pos_{},tgt_pos_{}; std::vector<EncoderLayer> enc_; std::vector<DecoderLayer> dec_; F32Vec final_{};
};

class Tokenizer {
public:
    explicit Tokenizer(const Model& m);
    std::vector<int> encode(const std::string& utf8,size_t max_tokens) const;
    std::string decode(const std::vector<int>& ids) const;
    bool target_allowed(int id) const;
private:
    const Model& m_;
    std::vector<std::vector<int>> buckets_;
    std::vector<int> target_ids_;
public:
    const std::vector<int>& target_ids() const { return target_ids_; }
};

struct Stats { uint32_t src_tokens=0,out_tokens=0; uint64_t encoder_us=0,decoder_us=0,total_us=0; };
class Translator {
public:
    explicit Translator(const Model& m);
    bool translate(const std::string& ru,std::string& en,Stats* stats=nullptr,std::string* error=nullptr);
private:
    const Model& m_; Tokenizer tok_; std::vector<int8_t> qscratch_; std::vector<float> fscratch_;
    void matvec(const Q8Tensor&,const float*,float*); void row(const Q8Tensor&,uint32_t,float*); void rms(float*,const float*,const F32Vec&); void ffn(float*,const F32Vec&,const Q8Tensor&,const Q8Tensor&);
    void encoder_attention(float* x,size_t S,const AttentionWeights&);
};
uint32_t crc32(const uint8_t* data,size_t n);
}
