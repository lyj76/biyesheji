%% EVM
function ErrorVecrms = EVMrms(DataMes, ConsIdeal)
DataMesNor = DataMes(:)/sqrt(mean(abs(DataMes(:)).^2));
ConsIdealNor = ConsIdeal(:)/sqrt(mean(abs(ConsIdeal(:)).^2));
ErrorVec = mean((abs(DataMesNor - ConsIdealNor)).^2);
ErrorVecrms= sqrt(ErrorVec);
end
