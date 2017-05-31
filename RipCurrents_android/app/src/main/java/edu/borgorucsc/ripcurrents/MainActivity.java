package edu.borgorucsc.ripcurrents;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        callOpenCV();
    }

    static{

        System.loadLibrary("myNative");
    }

    public native void callOpenCV();
}
