#include "nanomt.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#ifdef ESP_PLATFORM
#include "esp_timer.h"
#endif

namespace nanomt {
namespace {
constexpr size_t HDR=128; constexpr char MAGIC[8]={'S','3','M','T','0','0','0','1'};
static inline size_t a16(size_t x){return (x+15u)&~size_t(15u);}
struct Cursor { const uint8_t* p; const uint8_t* end; bool ok=true;
    uint32_t u32(){ if(p+4>end){ok=false;return 0;} uint32_t v; std::memcpy(&v,p,4); p+=4; return v; }
    uint16_t u16(){ if(p+2>end){ok=false;return 0;} uint16_t v; std::memcpy(&v,p,2); p+=2; return v; }
    bool skip_align(const uint8_t* base){ size_t off=size_t(p-base); size_t no=a16(off); if(base+no>end){ok=false;return false;} p=base+no; return true; }
};
static bool parse_f32(Cursor& c,const uint8_t* base,F32Vec& v,uint32_t expect){ uint32_t n=c.u32(); c.u32();c.u32();c.u32(); if(!c.ok||n!=expect||c.p+size_t(n)*4>c.end)return false; v.n=n; v.data=reinterpret_cast<const float*>(c.p); c.p+=size_t(n)*4; return c.skip_align(base); }
static bool parse_q8(Cursor& c,const uint8_t* base,Q8Tensor& t,uint32_t er,uint32_t ec){ uint32_t r=c.u32(),co=c.u32();c.u32();c.u32(); if(!c.ok||r!=er||co!=ec||c.p+size_t(r)*4>c.end)return false; t.rows=r;t.cols=co;t.scales=reinterpret_cast<const float*>(c.p);c.p+=size_t(r)*4;c.skip_align(base); if(!c.ok||c.p+size_t(r)*co>c.end)return false; t.q=reinterpret_cast<const int8_t*>(c.p); c.p+=size_t(r)*co; return c.skip_align(base); }
static inline uint64_t now_us(){
#ifdef ESP_PLATFORM
    return (uint64_t)esp_timer_get_time();
#else
    using namespace std::chrono; return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
#endif
}

static int32_t dot_i8(const int8_t* a,const int8_t* b,int n){
#if defined(ESP_PLATFORM) && defined(CONFIG_IDF_TARGET_ESP32S3)
    int32_t total=0; int blocks=n/16; const int8_t* pa=a; const int8_t* pb=b;
    if(blocks>0 && ((reinterpret_cast<uintptr_t>(pa)|reinterpret_cast<uintptr_t>(pb))&15u)==0){
        int32_t acc=0; int count=blocks*16;
        asm volatile(
            "EE.ZERO.ACCX\n"
            "1:\n"
            "EE.VLD.128.IP q0, %[pa], 16\n"
            "EE.VLD.128.IP q1, %[pb], 16\n"
            "EE.VMULAS.S8.ACCX q0, q1\n"
            "addi %[count], %[count], -16\n"
            "bnez %[count], 1b\n"
            "RUR.ACCX_0 %[acc]\n"
            : [acc] "=r"(acc), [pa] "+r"(pa), [pb] "+r"(pb), [count] "+r"(count) :: "memory");
        total+=acc;
        for(int i=blocks*16;i<n;i++) total+=int32_t(a[i])*int32_t(b[i]);
        return total;
    }
#endif
    int32_t s=0; for(int i=0;i<n;i++) s+=int32_t(a[i])*int32_t(b[i]); return s;
}
}

uint32_t crc32(const uint8_t* data,size_t n){ uint32_t c=0xffffffffu; for(size_t i=0;i<n;i++){ c^=data[i]; for(int k=0;k<8;k++) c=(c>>1)^(0xedb88320u & (0u-(c&1u))); } return ~c; }

bool Model::load(const void* data,size_t size,std::string* err){
    auto fail=[&](const char* s){if(err)*err=s;return false;}; if(!data||size<HDR)return fail("file too small"); const uint8_t* b=(const uint8_t*)data;
    if(std::memcmp(b,MAGIC,8)!=0) return fail("bad magic");
    uint32_t h[18]; std::memcpy(h,b+8,sizeof(h));
    if(h[0]!=1) return fail("unsupported version");
    cfg_={h[2],h[3],h[4],h[5],h[6],h[7],h[8],h[9],h[10],h[11],h[12],h[13]};
    if(cfg_.vocab < 260 || cfg_.vocab > 8192) return fail("invalid vocab size");
    if(cfg_.d_model < 16 || cfg_.d_model > 512 || (cfg_.d_model % 16) != 0) return fail("invalid d_model");
    if(cfg_.d_ff < cfg_.d_model || cfg_.d_ff > 4096 || (cfg_.d_ff % 16) != 0) return fail("invalid d_ff");
    if(cfg_.n_heads == 0 || cfg_.n_heads > 16 || (cfg_.d_model % cfg_.n_heads) != 0) return fail("invalid head configuration");
    if(cfg_.n_encoder == 0 || cfg_.n_encoder > 16 || cfg_.n_decoder == 0 || cfg_.n_decoder > 16) return fail("invalid layer count");
    if(cfg_.max_src < 4 || cfg_.max_src > 512 || cfg_.max_tgt < 4 || cfg_.max_tgt > 512) return fail("invalid sequence limits");
    if(cfg_.pad >= cfg_.vocab || cfg_.bos >= cfg_.vocab || cfg_.eos >= cfg_.vocab || cfg_.unk >= cfg_.vocab) return fail("invalid special token ids");
    uint32_t tok_bytes=h[14], payload_bytes=h[15], want_crc=h[16], total=h[17];
    if(total!=size || payload_bytes!=size-HDR || tok_bytes>payload_bytes) return fail("size mismatch");
    if(crc32(b+HDR,payload_bytes)!=want_crc) return fail("CRC32 mismatch");
    const uint8_t* base=b+HDR; Cursor c{base,b+size,true}; uint32_t v=c.u32(); if(v!=cfg_.vocab)return fail("tokenizer vocab mismatch"); pieces_.clear();pieces_.reserve(v);
    for(uint32_t i=0;i<v;i++){ uint16_t n=c.u16(); if(!c.ok||c.p+n>c.end)return fail("bad tokenizer"); pieces_.push_back({c.p,n});c.p+=n; }
    c.skip_align(base); if(!c.ok||size_t(c.p-base)!=tok_bytes)return fail("tokenizer size mismatch");
    if(!parse_q8(c,base,tok_,cfg_.vocab,cfg_.d_model)||!parse_q8(c,base,src_pos_,cfg_.max_src,cfg_.d_model)||!parse_q8(c,base,tgt_pos_,cfg_.max_tgt,cfg_.d_model))return fail("bad embeddings");
    enc_.assign(cfg_.n_encoder,{}); for(auto& l:enc_){ if(!parse_f32(c,base,l.n1,cfg_.d_model))return fail("bad enc norm1"); for(Q8Tensor* t:{&l.attn.q,&l.attn.k,&l.attn.v,&l.attn.o})if(!parse_q8(c,base,*t,cfg_.d_model,cfg_.d_model))return fail("bad enc attention"); if(!parse_f32(c,base,l.n2,cfg_.d_model)||!parse_q8(c,base,l.w1,cfg_.d_ff,cfg_.d_model)||!parse_q8(c,base,l.w2,cfg_.d_model,cfg_.d_ff))return fail("bad enc ffn"); }
    dec_.assign(cfg_.n_decoder,{}); for(auto& l:dec_){ if(!parse_f32(c,base,l.n1,cfg_.d_model))return fail("bad dec norm1"); for(Q8Tensor* t:{&l.self_attn.q,&l.self_attn.k,&l.self_attn.v,&l.self_attn.o})if(!parse_q8(c,base,*t,cfg_.d_model,cfg_.d_model))return fail("bad self attention"); if(!parse_f32(c,base,l.n2,cfg_.d_model))return fail("bad dec norm2"); for(Q8Tensor* t:{&l.cross_attn.q,&l.cross_attn.k,&l.cross_attn.v,&l.cross_attn.o})if(!parse_q8(c,base,*t,cfg_.d_model,cfg_.d_model))return fail("bad cross attention"); if(!parse_f32(c,base,l.n3,cfg_.d_model)||!parse_q8(c,base,l.w1,cfg_.d_ff,cfg_.d_model)||!parse_q8(c,base,l.w2,cfg_.d_model,cfg_.d_ff))return fail("bad dec ffn"); }
    if(!parse_f32(c,base,final_,cfg_.d_model)) return fail("bad final norm");
    if(c.p!=c.end) return fail("trailing/corrupt payload");
    return true;
}

Tokenizer::Tokenizer(const Model& m):m_(m),buckets_(256){
    const auto& ps=m_.pieces();
    target_ids_.reserve(ps.size());
    for(size_t id=4;id<ps.size();++id){
        const Piece& p=ps[id];
        if(p.len) buckets_[p.data[0]].push_back((int)id);
        bool ok=p.len>0;
        for(uint16_t j=0;j<p.len && ok;++j){ const uint8_t c=p.data[j]; if(c<0x20||c>0x7e) ok=false; }
        if(ok) target_ids_.push_back((int)id);
    }
    target_ids_.push_back((int)m_.config().eos);
    for(auto& b:buckets_) std::sort(b.begin(),b.end(),[&](int a,int z){
        const auto& pa=ps[a]; const auto& pz=ps[z];
        return pa.len!=pz.len ? pa.len>pz.len : a<z;
    });
}

std::vector<int> Tokenizer::encode(const std::string& s,size_t max_tokens) const{
    std::vector<int> out; out.reserve(max_tokens); out.push_back((int)m_.config().bos); const auto& ps=m_.pieces(); const auto* d=(const uint8_t*)s.data(); size_t N=s.size(),i=0;
    while(i<N && out.size()+1<max_tokens){ int best=-1; size_t bl=0; for(int id:buckets_[d[i]]){ auto p=ps[id]; if(p.len<=bl||i+p.len>N)continue; if(std::memcmp(p.data,d+i,p.len)==0){best=id;bl=p.len;break;} } if(best<0){best=4+d[i];bl=1;} out.push_back(best);i+=bl; }
    out.push_back((int)m_.config().eos); return out;
}
std::string Tokenizer::decode(const std::vector<int>& ids) const{ std::string s; for(int id:ids){ if(id==(int)m_.config().eos)break; if(id<4||id>=(int)m_.pieces().size())continue; auto p=m_.pieces()[id]; s.append((const char*)p.data,p.len); } return s; }
bool Tokenizer::target_allowed(int id) const{ if(id==(int)m_.config().eos)return true; if(id<4||id>=(int)m_.pieces().size())return false; auto p=m_.pieces()[id]; if(!p.len)return false; for(uint16_t i=0;i<p.len;i++){ uint8_t c=p.data[i]; if(c<0x20||c>0x7e)return false; } return true; }

Translator::Translator(const Model& m):m_(m),tok_(m){ qscratch_.resize(std::max(m.config().d_model,m.config().d_ff)+16); fscratch_.resize(std::max(m.config().d_model,m.config().d_ff)*8); }
void Translator::row(const Q8Tensor&t,uint32_t r,float*out){ const int8_t* q=t.q+size_t(r)*t.cols; float s=t.scales[r]; for(uint32_t i=0;i<t.cols;i++)out[i]=q[i]*s; }
void Translator::matvec(const Q8Tensor&t,const float*x,float*out){ float mx=0; for(uint32_t i=0;i<t.cols;i++)mx=std::max(mx,std::fabs(x[i])); float as=mx>0?mx/127.f:1.f; uintptr_t raw=(uintptr_t)qscratch_.data(); int8_t* aq=(int8_t*)((raw+15u)&~uintptr_t(15u)); for(uint32_t i=0;i<t.cols;i++)aq[i]=(int8_t)std::max(-127,std::min(127,(int)std::lrint(x[i]/as))); for(uint32_t r=0;r<t.rows;r++){ const int8_t* w=t.q+size_t(r)*t.cols; out[r]=float(dot_i8(aq,w,(int)t.cols))*as*t.scales[r]; } }
void Translator::rms(float*out,const float*x,const F32Vec&w){ float ss=0;for(uint32_t i=0;i<w.n;i++)ss+=x[i]*x[i]; float inv=1.f/std::sqrt(ss/w.n+1e-5f);for(uint32_t i=0;i<w.n;i++)out[i]=x[i]*inv*w.data[i]; }
void Translator::ffn(float*x,const F32Vec& n,const Q8Tensor&w1,const Q8Tensor&w2){ uint32_t d=m_.config().d_model,ff=m_.config().d_ff; float* z=fscratch_.data(); float* h=z+d; rms(z,x,n); matvec(w1,z,h); for(uint32_t i=0;i<ff;i++)h[i]=h[i]/(1.f+std::exp(-h[i])); float* y=h+ff; matvec(w2,h,y); for(uint32_t i=0;i<d;i++)x[i]+=y[i]; }
void Translator::encoder_attention(float*x,size_t S,const AttentionWeights&w){ uint32_t d=m_.config().d_model,H=m_.config().n_heads,hd=d/H; std::vector<float> q(S*d),k(S*d),v(S*d),ctx(d),y(d); for(size_t t=0;t<S;t++){matvec(w.q,x+t*d,q.data()+t*d);matvec(w.k,x+t*d,k.data()+t*d);matvec(w.v,x+t*d,v.data()+t*d);} float scale=1.f/std::sqrt((float)hd); std::vector<float> sc(S);
    for(size_t t=0;t<S;t++){std::fill(ctx.begin(),ctx.end(),0.f);for(uint32_t h=0;h<H;h++){float mx=-1e30f,sum=0;for(size_t j=0;j<S;j++){float z=0;for(uint32_t u=0;u<hd;u++)z+=q[t*d+h*hd+u]*k[j*d+h*hd+u];sc[j]=z*scale;mx=std::max(mx,sc[j]);}for(size_t j=0;j<S;j++){sc[j]=std::exp(sc[j]-mx);sum+=sc[j];}for(size_t j=0;j<S;j++){float a=sc[j]/sum;for(uint32_t u=0;u<hd;u++)ctx[h*hd+u]+=a*v[j*d+h*hd+u];}}matvec(w.o,ctx.data(),y.data());for(uint32_t u=0;u<d;u++)x[t*d+u]+=y[u];}
}

bool Translator::translate(const std::string& ru,std::string& en,Stats* stats,std::string* err){ auto fail=[&](const char*s){if(err)*err=s;return false;}; const auto& c=m_.config(); auto ids=tok_.encode(ru,c.max_src+1); if(ids.size()>c.max_src)return fail("source too long"); size_t S=ids.size(); uint64_t all0=now_us(),e0=all0; std::vector<float> enc(S*c.d_model),norm(S*c.d_model);
    for(size_t t=0;t<S;t++){row(m_.token_embedding(),ids[t],enc.data()+t*c.d_model);std::vector<float> p(c.d_model);row(m_.src_pos(),t,p.data());for(uint32_t i=0;i<c.d_model;i++)enc[t*c.d_model+i]+=p[i];}
    for(const auto& l:m_.enc()){
        for(size_t t=0;t<S;t++)rms(norm.data()+t*c.d_model,enc.data()+t*c.d_model,l.n1);
        std::vector<float> before=norm; encoder_attention(norm.data(),S,l.attn); for(size_t i=0;i<enc.size();i++)enc[i]+=norm[i]-before[i];
        for(size_t t=0;t<S;t++)ffn(enc.data()+t*c.d_model,l.n2,l.w1,l.w2);
    }
    uint64_t e1=now_us(),d0=e1;
    struct Cache{std::vector<float> sk,sv,ck,cv;}; std::vector<Cache> cache(c.n_decoder);
    for(uint32_t L=0;L<c.n_decoder;L++){auto& ca=cache[L];ca.sk.resize(size_t(c.max_tgt)*c.d_model);ca.sv.resize(size_t(c.max_tgt)*c.d_model);ca.ck.resize(S*c.d_model);ca.cv.resize(S*c.d_model);const auto& l=m_.dec()[L];for(size_t t=0;t<S;t++){matvec(l.cross_attn.k,enc.data()+t*c.d_model,ca.ck.data()+t*c.d_model);matvec(l.cross_attn.v,enc.data()+t*c.d_model,ca.cv.data()+t*c.d_model);}}
    std::vector<int> outids; outids.reserve(c.max_tgt); int token=c.bos; uint32_t d=c.d_model,H=c.n_heads,hd=d/H; std::vector<float> x(d),z(d),q(d),k(d),v(d),ctx(d),proj(d),scores(std::max<size_t>(S,c.max_tgt)); float atscale=1.f/std::sqrt((float)hd);
    for(uint32_t step=0;step+1<c.max_tgt;step++){
        row(m_.token_embedding(),token,x.data()); row(m_.tgt_pos(),step,z.data());for(uint32_t i=0;i<d;i++)x[i]+=z[i];
        for(uint32_t L=0;L<c.n_decoder;L++){const auto& l=m_.dec()[L];auto& ca=cache[L];
            rms(z.data(),x.data(),l.n1);matvec(l.self_attn.q,z.data(),q.data());matvec(l.self_attn.k,z.data(),k.data());matvec(l.self_attn.v,z.data(),v.data());std::memcpy(ca.sk.data()+size_t(step)*d,k.data(),d*4);std::memcpy(ca.sv.data()+size_t(step)*d,v.data(),d*4);std::fill(ctx.begin(),ctx.end(),0.f);
            for(uint32_t h=0;h<H;h++){float mx=-1e30f,sum=0;for(uint32_t j=0;j<=step;j++){float s=0;for(uint32_t u=0;u<hd;u++)s+=q[h*hd+u]*ca.sk[size_t(j)*d+h*hd+u];scores[j]=s*atscale;mx=std::max(mx,scores[j]);}for(uint32_t j=0;j<=step;j++){scores[j]=std::exp(scores[j]-mx);sum+=scores[j];}for(uint32_t j=0;j<=step;j++){float a=scores[j]/sum;for(uint32_t u=0;u<hd;u++)ctx[h*hd+u]+=a*ca.sv[size_t(j)*d+h*hd+u];}}
            matvec(l.self_attn.o,ctx.data(),proj.data());for(uint32_t i=0;i<d;i++)x[i]+=proj[i];
            rms(z.data(),x.data(),l.n2);matvec(l.cross_attn.q,z.data(),q.data());std::fill(ctx.begin(),ctx.end(),0.f);
            for(uint32_t h=0;h<H;h++){float mx=-1e30f,sum=0;for(size_t j=0;j<S;j++){float s=0;for(uint32_t u=0;u<hd;u++)s+=q[h*hd+u]*ca.ck[j*d+h*hd+u];scores[j]=s*atscale;mx=std::max(mx,scores[j]);}for(size_t j=0;j<S;j++){scores[j]=std::exp(scores[j]-mx);sum+=scores[j];}for(size_t j=0;j<S;j++){float a=scores[j]/sum;for(uint32_t u=0;u<hd;u++)ctx[h*hd+u]+=a*ca.cv[j*d+h*hd+u];}}
            matvec(l.cross_attn.o,ctx.data(),proj.data());for(uint32_t i=0;i<d;i++)x[i]+=proj[i];ffn(x.data(),l.n3,l.w1,l.w2);
        }
        rms(z.data(),x.data(),m_.final_norm()); float mx=0;for(float a:z)mx=std::max(mx,std::fabs(a));float as=mx>0?mx/127.f:1.f;uintptr_t raw=(uintptr_t)qscratch_.data();int8_t* aq=(int8_t*)((raw+15u)&~uintptr_t(15u));for(uint32_t i=0;i<d;i++)aq[i]=(int8_t)std::max(-127,std::min(127,(int)std::lrint(z[i]/as)));int best=c.eos;float bests=-std::numeric_limits<float>::infinity();float oscale=1.f/std::sqrt((float)d);
        for(int id:tok_.target_ids()){const int8_t* w=m_.token_embedding().q+size_t(id)*d;float s=float(dot_i8(aq,w,d))*as*m_.token_embedding().scales[id]*oscale;if(s>bests){bests=s;best=id;}}
        token=best;if(token==(int)c.eos)break;outids.push_back(token);
    }
    uint64_t d1=now_us(); en=tok_.decode(outids); if(stats){stats->src_tokens=S;stats->out_tokens=outids.size();stats->encoder_us=e1-e0;stats->decoder_us=d1-d0;stats->total_us=d1-all0;} return true;
}
}
