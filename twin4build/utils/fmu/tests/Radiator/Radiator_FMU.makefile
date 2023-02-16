# FIXME: before you push into master...
RUNTIMEDIR=C:/Program Files/OpenModelica1.18.0-64bit/include/omc/c/
#COPY_RUNTIMEFILES=$(FMI_ME_OBJS:%= && (OMCFILE=% && cp $(RUNTIMEDIR)/$$OMCFILE.c $$OMCFILE.c))

fmu:
	rm -f Radiator.fmutmp/sources/Radiator_init.xml
	cp -a "C:/Program Files/OpenModelica1.18.0-64bit/share/omc/runtime/c/fmi/buildproject/"* Radiator.fmutmp/sources
	cp -a Radiator_FMU.libs Radiator.fmutmp/sources/

