./sdnn_stream --nn 1024 --nl 120;
./sdnn_stream --nn 4096 --nl 120;
./sdnn_stream --nn 16384 --nl 120;


./sdnn_stream --nn 1024 --nl 480;
./sdnn_stream --nn 4096 --nl 480;
./sdnn_stream --nn 16384 --nl 480;


./sdnn_stream --nn 1024 --nl 1920;
./sdnn_stream --nn 4096 --nl 1920;
./sdnn_stream --nn 16384 --nl 1920;

./sdnn_stream --nn 65536  --nl 120;
./sdnn_stream --nn 65536  --nl 1920;
./sdnn_stream --nn 65536  --nl 480;

./sdnn_cusparse --nn 1024 --nl 120;
./sdnn_cusparse --nn 4096 --nl 120;
./sdnn_cusparse --nn 16384 --nl 120;
./sdnn_cusparse --nn 65536 --nl 120;

./sdnn_cusparse --nn 1024 --nl 480;
./sdnn_cusparse --nn 4096 --nl 480;
./sdnn_cusparse --nn 16384 --nl 480;


./sdnn_cusparse --nn 1024 --nl 1920;
./sdnn_cusparse --nn 4096 --nl 1920;
./sdnn_cusparse --nn 16384 --nl 1920;

./sdnn_cusparse --nn 65536 --nl 1920;
./sdnn_cusparse --nn 65536 --nl 480;