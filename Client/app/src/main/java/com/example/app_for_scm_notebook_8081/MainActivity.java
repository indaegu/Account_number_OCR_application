package com.example.app_for_scm_notebook_8081;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.Socket;
import android.util.Log;

public class MainActivity extends AppCompatActivity {
    // 권한 및 이미지 캡쳐를 위한 요청 코드
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_CAMERA_PERMISSION = 2;

    // UI 컴포넌트들
    private EditText editTextBankName;
    private EditText editTextAccountNumber;
    private EditText editTextAccountHolder;
    private Button captureButton;
    private Button transferButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // UI 컴포넌트 초기화
        editTextBankName = findViewById(R.id.editText_bankName);
        editTextAccountNumber = findViewById(R.id.editText_accountNumber);
        editTextAccountHolder = findViewById(R.id.editText_accountHolder);
        captureButton = findViewById(R.id.button_capture);
        transferButton = findViewById(R.id.button_transfer);

        // 사진 촬영 버튼 클릭 이벤트 리스너 설정
        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Android M 버전 이상일 경우 권한 확인
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
                    } else {
                        dispatchTakePictureIntent();
                    }
                } else {
                    dispatchTakePictureIntent();
                }
            }
        });

        // 송금 버튼 클릭 이벤트 리스너 설정
        transferButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                transfer();
            }
        });
    }

    // 카메라로 사진을 촬영하기 위한 인텐트 생성 및 실행
    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    // 카메라 권한 요청 결과 처리
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                dispatchTakePictureIntent();
            } else {
                Toast.makeText(this, "카메라 권한이 필요합니다.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // 카메라로 촬영한 결과 처리
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            sendPhotoUsingTcpSocket(photo);
        }
    }

    // TCP 소켓을 사용하여 서버에 사진 전송
    private void sendPhotoUsingTcpSocket(final Bitmap photo) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                Socket socket = null;
                try {
                    // 사진 촬영 후 전송 시작 시점의 시간을 측정

                    long startTime = System.currentTimeMillis();
                    System.out.println("Start Time : " + startTime);

                    String serverIp = "192.168.152.177"; // 서버 ip주소 입력
                    int serverPort = 46460;
                    socket = new Socket(serverIp, serverPort);
                    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                    photo.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
                    byte[] photoBytes = byteArrayOutputStream.toByteArray();
                    DataOutputStream dataOutputStream = new DataOutputStream(socket.getOutputStream());
                    DataInputStream dataInputStream = new DataInputStream(socket.getInputStream());
                    dataOutputStream.writeInt(photoBytes.length);
                    dataOutputStream.write(photoBytes);
                    dataOutputStream.flush();
                    BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                    StringBuilder dataReceived = new StringBuilder();
                    String line;
                    while ((line = in.readLine()) != null) {
                        dataReceived.append(line);
                    }

                    // 받은 문자열 데이터를 튜플 형태로 변환
                    String[] dataTuple = dataReceived.toString().split(",");
                    System.out.println(dataTuple.length);
                    // 출력
                    if (dataTuple.length == 3) {
                        final String accountNumber = dataTuple[0].trim();
                        final String bankName = dataTuple[1].trim();
                        final String accountHolder = dataTuple[2].trim();

                        // UI 스레드에서 TextView 업데이트
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                editTextBankName.setText(bankName);
                                editTextAccountNumber.setText(accountNumber);
                                editTextAccountHolder.setText(accountHolder);
                            }
                        });
                    } else if (dataTuple.length == 2) {
                        // 길이가 2인 경우에 대한 처리
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                                builder.setTitle("경고");
                                builder.setMessage("올바르지 않은 계좌 형식입니다.\n재촬영 해주세요!");
                                builder.setPositiveButton("확인", new DialogInterface.OnClickListener() {
                                    @Override
                                    public void onClick(DialogInterface dialog, int which) {
                                        captureButton.setVisibility(View.VISIBLE);
                                    }
                                });
                                AlertDialog alertDialog = builder.create();
                                alertDialog.show();
                            }
                        });
                    } else {
                        // 잘못된 계좌 정보를 받았을 때 경고 메시지를 띄우는 코드
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                                builder.setTitle("경고");
                                builder.setMessage("DB에 존재하지 않는 계좌입니다.\n재촬영 해주세요!");
                                builder.setPositiveButton("확인", new DialogInterface.OnClickListener() {
                                    @Override
                                    public void onClick(DialogInterface dialog, int which) {
                                        captureButton.setVisibility(View.VISIBLE);
                                    }
                                });
                                AlertDialog alertDialog = builder.create();
                                alertDialog.show();
                            }
                        });
                    }
                    // 서버로부터 응답을 받은 시점의 시간을 측정
                    long endTime = System.currentTimeMillis();

                    // 소요된 시간을 계산
                    long elapsedTime = endTime - startTime;

                    // 소요된 시간을 로그에 출력
                    System.out.println("Elapsed Time: " + elapsedTime + " ms");
                    Log.d("MyApp", "실행시간: " + elapsedTime + "ms");


                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    if (socket != null) {
                        try {
                            socket.close();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }).start();
    }

    // 송금 실행 메소드
    private void transfer() {
        String bankName = editTextBankName.getText().toString();
        String accountNumber = editTextAccountNumber.getText().toString();
        String accountHolder = editTextAccountHolder.getText().toString();

        // 모든 필드가 입력되어 있을 경우 송금 실행
        if (!bankName.isEmpty() && !accountNumber.isEmpty() && !accountHolder.isEmpty()) {
            // 실제 송금 로직이 들어갈 부분
            Toast.makeText(this, "송금이 완료되었습니다.", Toast.LENGTH_SHORT).show();
            editTextBankName.setText("");
            editTextAccountNumber.setText("");
            editTextAccountHolder.setText("");
        } else {
            Toast.makeText(this, "모든 필드를 입력해 주세요.", Toast.LENGTH_SHORT).show();
        }
    }
}



