package app.ij.mlwithtensorflowlite;
import org.jetbrains.annotations.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.media.MediaPlayer;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import app.ij.mlwithtensorflowlite.ml.ModelUnquant;

public class MainActivity extends AppCompatActivity {
    TextView result, confidence;ImageView imageView;
    Button picture;
    int imageSize = 224;
    private static final float MIN_CONFIDENCE_THRESHOLD = 0.5f;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }
    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);
            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
    public void classifyImage(Bitmap image) {
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            // Crea inputs para la inferencia del modelo
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            // Ejecuta la inferencia del modelo y obtén el resultado
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();


            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;

            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"veinte", "cien","otra denominacion"};
            result.setText(classes[maxPos]);
            playAudio(classes[maxPos]);
//codigo de prueba//

            if (maxConfidence >= MIN_CONFIDENCE_THRESHOLD) {
                result.setText(classes[maxPos]);
                playAudio(classes[maxPos]);
            } else {
                handleUnrecognizedBill();
            }
            // Añade un bloque else para manejar el caso en que el billete no sea de 20 o 100 pesos
            if (!(result.getText().equals("veinte") || result.getText().equals("cien"))) {
                handleOtherBill();

            }
            String s = "";
//Fragmento codigo caso no ser billete de 20 o 100//
            int maxIndex = 0;

            for (int i = 1; i < confidences.length; i++) {
                if (confidences[i] > confidences[maxIndex]) {
                    maxIndex = i;
                }
            }

            s += String.format("%s: %.1f%%\n", classes[maxIndex], confidences[maxIndex] * 100);
            confidence.setText(s);
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
//hasta aca//////
    }

    private void playAudio(String className) {
        int audioResource;

        if ("veinte".equals(className)) {
            audioResource = R.raw.veintepesos;
        } else if ("cien".equals(className)) {
            audioResource = R.raw.cienpesos;
        } else {
            audioResource = R.raw.noreconocido;
        }
        // Reproduce el audio correspondiente
        final MediaPlayer mediaPlayer = MediaPlayer.create(this, audioResource);
        mediaPlayer.start();
    }
    private void handleUnrecognizedBill() {
        result.setText("Billete no reconocido");
        confidence.setText("");
        // Agrega aquí cualquier acción adicional que desees realizar para el caso de billete no reconocido
    }
    private void handleOtherBill() {
        result.setText("Billete no reconocido (Otra denominación)");
        confidence.setText("");
        // Agrega aquí cualquier acción adicional que desees realizar para el caso de otras denominaciones
    }
}



