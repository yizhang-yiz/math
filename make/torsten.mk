%_torsten__.stan: %.stan
	@echo "Generating Torsten input file $@ ..."
	@cp $< $@
	@touch $(basename $<)_torsten_functions__.stan
	@perl -0777 -pi -e 's/(^ *functions *{)/$$1\n#include $$ARGV[0]\n/gm' $@ ${CURDIR}/$(basename $<)_torsten_functions__.stan
	@perl -0777 -pi -e 's/(torsten_generalOdeModel_rk45)\n?\(\n? *([^,]+), */$$1_$$2__\(/gm' $@
	@perl -0777 -pi -e 's/(torsten_generalOdeModel_bdf)\n?\(\n? *([^,]+), */$$1_$$2__\(/gm' $@
	@rm $(basename $<)_torsten_functions__.stan

%_torsten_functions__.stan: %_torsten__.stan
	@echo "Generating Torsten functions file $@ ..."
	@perl -0777 -ne 'while(m/(torsten_generalOdeModel_rk45_[^\(]+)\(/g){print \
		 "matrix $$1(int nCmt, real[] time, real[] amt, real[] rate, real[] ii, int[] evid, int[] cmt, int[] addl, int[] ss, real[] theta, real[] biovar, real[] tlag, real rel_tol, real abs_tol, int max_step);\n"}' $< > $@
	@perl -0777 -ne 'while(m/(torsten_generalOdeModel_bdf_[^\(]+)\(/g){print \
		 "matrix $$1(int nCmt, real[] time, real[] amt, real[] rate, real[] ii, int[] evid, int[] cmt, int[] addl, int[] ss, real[] theta, real[] biovar, real[] tlag, real rel_tol, real abs_tol, int max_step);\n"}' $< >> $@

%.torsten.hpp: STANCFLAGS+=--allow_undefined
%.torsten.hpp: %_torsten__.stan %_torsten_functions__.stan $(MODEL_HEADER) bin/stanc$(EXE)
	@echo ''
	@echo '--- Torsten: Translating Stan model to C++ code ---'
	bin$(PATH_SEPARATOR)stanc$(EXE) $(STANCFLAGS) $< --o=$@
	@perl -0777 -pi -e 's/([\S]+ *torsten_generalOdeModel_rk45)_([^,]+)__\(/$$1\($$2_functor__(), /gm' $@
	@perl -0777 -pi -e 's/([\S]+ *torsten_generalOdeModel_bdf)_([^,]+)__\(/$$1\($$2_functor__(), /gm' $@

%.torsten$(EXE) : %.hpp.out %_torsten__.stan
	@echo ''
	@echo '--- Torsten: Linking C++ model ---'
	@test -f $(dir $<)USER_HEADER.hpp || touch $(dir $<)USER_HEADER.hpp
	$(LINK.cc) -O$O $(OUTPUT_OPTION) $(CMDSTAN_MAIN) -include $< -include $(dir $<)USER_HEADER.hpp $(LIBSUNDIALS)
