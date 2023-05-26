from ess import TraceWin, lib_tw

#file_name_lat = 'testfiles/lattice.dat'
tracewin_lattice = "E:\\PyCharmWorkspace\\TBSAC\\CAFeII_SC_Proton\\SC_Ca.dat"
lat = lib_tw.LATTICE(tracewin_lattice)
print(lat.lst)
for i in range(len(lat.lst)):
    if lat.lst[i].typ == 'FIELD_MAP':
        if i == 3:
            print(lat.lst[i].__dir__())
            print(lat.lst[i].Bnom)



