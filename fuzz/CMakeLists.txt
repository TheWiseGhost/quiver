add_executable(fuzz_hnsw_load fuzz_hnsw_load.cc)
target_link_libraries(fuzz_hnsw_load PRIVATE quiver)
target_compile_options(fuzz_hnsw_load PRIVATE -fsanitize=fuzzer,address,undefined)
target_link_options(fuzz_hnsw_load PRIVATE -fsanitize=fuzzer,address,undefined)
# Instrument the library too, so coverage guidance sees the parser.
target_compile_options(quiver PRIVATE -fsanitize=fuzzer-no-link,address,undefined)
target_link_options(quiver INTERFACE -fsanitize=address,undefined)

# Writes small valid index files to seed the fuzzer (random bytes never pass the header CRC).
add_executable(make_fuzz_corpus make_corpus.cc)
target_link_libraries(make_fuzz_corpus PRIVATE quiver)
target_compile_options(make_fuzz_corpus PRIVATE -fsanitize=address,undefined)
target_link_options(make_fuzz_corpus PRIVATE -fsanitize=address,undefined)
