@echo off
echo    GENERATOR RUCHU
echo.

echo 1/7: Generate regular cars...
python "%SUMO_HOME%\tools\randomTrips.py" -n network\osm.net.xml -o demand\car.trips.xml -r demand\car.rou.xml --prefix car_ --additional-files demand\vtypes.add.xml --vclass passenger --trip-attributes "type='car'" --validate --lanes --fringe-factor 5.0 --poisson -p 2 -e 3600

echo.
echo 2/7: Generate electric cars...
python "%SUMO_HOME%\tools\randomTrips.py" -n network\osm.net.xml -o demand\car_ev.trips.xml -r demand\car_ev.rou.xml --prefix car_ev_ --additional-files demand\vtypes.add.xml --vclass passenger --trip-attributes "type='car_ev'" --validate --lanes --fringe-factor 5.0 --poisson -p 25.0 -e 3600

echo.
echo 3/7: Generate trucks...
python "%SUMO_HOME%\tools\randomTrips.py" -n network\osm.net.xml -o demand\truck.trips.xml -r demand\truck.rou.xml --prefix truck_ --additional-files demand\vtypes.add.xml --vclass truck --trip-attributes "type='truck'" --validate --lanes --fringe-factor 5.0 --poisson -p 1800.0 -e 3600

echo.
echo 4/7: Generate buses...
python "%SUMO_HOME%\tools\randomTrips.py" -n network\osm.net.xml -o demand\bus.trips.xml -r demand\bus.rou.xml --prefix bus_ --additional-files demand\vtypes.add.xml --vclass bus --trip-attributes "type='bus'" --validate --lanes --fringe-factor 5.0 --poisson -p 50.0 -e 3600

echo.
echo 5/7: Generate motorcycles...
python "%SUMO_HOME%\tools\randomTrips.py" -n network\osm.net.xml -o demand\motorcycle.trips.xml -r demand\motorcycle.rou.xml --prefix motorcycle_ --additional-files demand\vtypes.add.xml --vclass motorcycle --trip-attributes "type='motorcycle'" --validate --lanes --fringe-factor 5.0 --poisson -p 120.0 -e 3600

echo.
echo 6/7: Generate trams...
python "%SUMO_HOME%\tools\randomTrips.py" -n network\osm.net.xml -o demand\tram.trips.xml -r demand\tram.rou.xml --prefix tram_ --additional-files demand\vtypes.add.xml --vclass tram --trip-attributes "type='tram_gdansk'" --validate --lanes --fringe-factor 5.0 --poisson -p 180.0 -e 3600

echo.
echo 7/7: Generate emergency vehicles...
python "%SUMO_HOME%\tools\randomTrips.py" -n network\osm.net.xml -o demand\emergency.trips.xml -r demand\emergency.rou.xml --prefix emergency_ --additional-files demand\vtypes.add.xml --vclass emergency --trip-attributes "type='emergency'" --validate --lanes --fringe-factor 5.0 --poisson -p 1800.0 -e 3600
pause
