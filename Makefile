CXX := dpcpp
CXXFLAGS = -O2 -g -std=c++17

SRC := src/parallel_bfs.cpp
USM_SRC := src/vector-add-usm.cpp

.PHONY: fpga_emu run_emu fpga_emu_usm run_emu_usm clean 

fpga_emu: parallel_bfs.fpga_emu

hw: parallel_bfs.fpga

report: parallel_bfs_report.a

parallel_bfs.fpga_emu:
	dpcpp -fintelfpga -DFPGA_EMULATOR src/host.cpp src/parallel_bfs_kernel.cpp -o emu_parallel -O0

a.o: $(SRC)
	$(CXX) $(CXXFLAGS) -fintelfpga -c $^ -o $@ -DFPGA=1

parallel_bfs.fpga: a.o
	$(CXX) $(CXXFLAGS) -fintelfpga $^ -o $@ -Xshardware -Xsprofile

run_emu: parallel_bfs.fpga_emu
	./parallel_bfs.fpga_emu


run_hw: parallel_bfs.fpga
	./parallel_bfs.fpga

dev.o: $(SRC)
	$(CXX) $(CXXFLAGS) -fintelfpga -c $^ -o $@ -DFPGA=1



parallel_bfs_report.a: dev.o
	$(CXX) $(CXXFLAGS) -fintelfpga -fsycl-link $^ -o $@ -Xshardware

clean:
	rm -rf *.o *.d *.out *.mon *.emu *.aocr *.aoco *.prj *.fpga_emu *.fpga_emu_buffers parallel_bfs.fpga  vector-add-usm.fpga *.a
