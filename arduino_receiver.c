#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <time.h>
#include <sys/time.h>
#include "arduino_receiver.h"

void get_arduino_data(const char* filename, const char* portname, time_t duration) {
    int serial_port = open(portname, O_RDWR);

    if(access(filename, F_OK) != -1) //File exists
    {
        if (remove(filename) == 0) {
            printf("file %s is removed\n",filename);
        } 
        else {
            perror("Error deleting file\n");
        }
    }

    FILE *file = fopen(filename, "ab");
    if (file == NULL) {
        
        perror("fopen");
        return;
    }

    struct timeval start_time;
    gettimeofday(&start_time, NULL);


    if (serial_port < 0) {
        perror("Error opening serial port");
        return;
    }

    struct termios tty;
    if (tcgetattr(serial_port, &tty) != 0) {
        perror("Error getting serial port attributes");
        close(serial_port);
        return;
    }

    cfsetospeed(&tty, B115200);  // Adjust the baud rate as needed
    cfsetispeed(&tty, B115200);

    tty.c_cflag |= (CLOCAL | CREAD);  // Enable receiver and ignore modem control lines

    // Set the character size
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;  // 8-bit characters

    // No parity
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;  // 1 stop bit

    // Raw input
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    // Read at least one character or block for 100 ms
    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
        perror("Error setting serial port attributes");
        close(serial_port);
        return;
    }

    // Read data from the serial port
    char buffer[1024];
    ssize_t bytesRead;
    
    time_t startTime = time(NULL);

    int count = 0;

    while (time(NULL) - startTime < duration) {
        bytesRead = read(serial_port, buffer, sizeof(buffer));
        if (bytesRead > 0) {

            count++;

            //Storing current time as epochs for timestamping
            long long t = time(NULL);

            float left_omega, right_omega, angle;

            //Parse and store variables from the UART buffer
            sscanf(buffer, "L: %f R: %f A:%f\n", &left_omega, &right_omega, &angle) == 2;

            //Write values to the file with timestamp
            fwrite(&t, sizeof(long long), 1, file);
            fwrite(&right_omega, sizeof(double), 1, file);
            fwrite(&left_omega, sizeof(double), 1, file);
            fwrite(&angle, sizeof(double), 1, file);
        } 
        else if (bytesRead < 0) 
        {
            perror("Error reading from serial port");
            break;
        }
    }

    // Close the serial port
    close(serial_port);
    fclose(file);

    //Print number of datapoints read
    printf("Count = %d", count);

}

// int main() {
//     const char* portname = "/dev/ttyACM0";  // Adjust this based on your COM port

    

//     time_t duration = 20;

    

//     return 0;
// }