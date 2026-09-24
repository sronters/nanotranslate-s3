#include "nanomt.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
int main(int argc,char**argv){ if(argc<2){std::cerr<<"usage: nanomt_cli model.s3mt [Russian sentence]\n";return 2;} std::ifstream f(argv[1],std::ios::binary);std::vector<char>b((std::istreambuf_iterator<char>(f)),{});nanomt::Model m;std::string e;if(!m.load(b.data(),b.size(),&e)){std::cerr<<"load: "<<e<<"\n";return 1;}nanomt::Translator tr(m);auto run=[&](const std::string&s){std::string out;nanomt::Stats st;if(!tr.translate(s,out,&st,&e)){std::cerr<<"error: "<<e<<"\n";return;}std::cout<<out<<"\n["<<st.total_us/1000.0<<" ms, src="<<st.src_tokens<<", out="<<st.out_tokens<<"]\n";};if(argc>2){std::string s=argv[2];for(int i=3;i<argc;i++)s+=" "+std::string(argv[i]);run(s);return 0;}std::string s;while(std::cout<<"ru> "&&std::getline(std::cin,s))if(!s.empty())run(s);}
