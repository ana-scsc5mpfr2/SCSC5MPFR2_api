from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import logging
import traceback
import matplotlib.colors as mcolors

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
MAX_SIZE = 2024
V = 0 #15
N = 2
U = 0 #12
M = 0 #U - N

def validate_image(image):
    """Validate image size and format"""
    if image.size[0] > MAX_SIZE or image.size[1] > MAX_SIZE:
        raise ValueError(f"Image dimensions must not exceed {MAX_SIZE}x{MAX_SIZE} pixels")

    allowed_formats = {'PNG', 'JPEG', 'JPG'}
    if image.format.upper() not in allowed_formats:
        raise ValueError(f"Invalid image format. Allowed formats: {', '.join(allowed_formats)}")

def encontrar_tres_picos_mas_altos(hist):
    """Encuentra los tres picos más altos excluyendo el color blanco."""
    picos = np.argsort(hist.flatten())[-3:]
    picos_filtrados = [pico for pico in picos if hist[pico] < np.max(hist) * 0.9]
    return picos_filtrados[:2]  # Devolver los dos picos más altos que no son blancos

def calcular_histograma_hue(image):
    """Calculate hue histogram without using OpenCV."""
    hist = np.zeros(180)
    pixels = np.array(image.convert('HSV'))  # Convert to HSV using PIL
    for pixel in pixels.reshape(-1, 3):
        hue = pixel[0]
        if 0 <= hue < 180:  # Asegurarse de que hue esté dentro del rango
            hist[hue] += 1
    return hist

def mostrar_histograma_con_picos(hist):
    """Encuentra los picos más altos y aplica suavizado al histograma."""
    picos = encontrar_tres_picos_mas_altos(hist)
    return picos

def mostrar_imagen_original_y_colores(imagen_array, picos):
    """Muestra la imagen original y convierte los picos a valores de tono."""
    imagen_hsv = np.array(imagen_array.convert('HSV'))
    tonos_picos = [pico / 180.0 for pico in picos]
    return tonos_picos

def redimensionar_imagen(imagen_array, tamano):
    """Redimensiona la imagen al tamaño especificado."""
    return imagen_array.resize((tamano, tamano))

def redimensionar_imagen_cuadrada(imagen_array):
    """Redimensiona la imagen al tamaño de un cuadrado."""
    ancho, altura = imagen_array.size
    tamano_maximo = max(ancho, altura)
    nueva_imagen = Image.new("RGB", (tamano_maximo, tamano_maximo))
    nueva_imagen.paste(imagen_array, (0, 0))
    return nueva_imagen

def dibujar_circulos_concentricos(imagen, centro, radios):
    """Dibuja círculos concéntricos alrededor del centro dado."""
    draw = ImageDraw.Draw(imagen)
    for radio in radios:
        draw.ellipse((centro[0] - radio, centro[1] - radio, centro[0] + radio, centro[1] + radio), outline="green", width=2)
    return imagen

def dibujar_lineas_radiales(image, lower_bound, upper_bound, num_lines=360, color_linea="red"):
    """Dibuja líneas radiales desde el centro de la imagen."""
    width, height = image.size
    center = (width // 2, height // 2)
    radio_imagen = min(width, height) / 2
    resultados = []
    draw = ImageDraw.Draw(image)
    imagen_hsv = np.array(image.convert('HSV'))

    for i in range(num_lines):
        angle = 2 * np.pi * i / num_lines + np.pi
        angle_deg = np.degrees(angle) % 360
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        for r in range(int(radio_imagen)):
            x = int(center[0] + cos_angle * r)
            y = int(center[1] - sin_angle * r)
            if 0 <= x < width and 0 <= y < height:
                pixel_color = imagen_hsv[y, x]
                if lower_bound[0] <= pixel_color[0] <= upper_bound[0] and lower_bound[1] <= pixel_color[1] <= upper_bound[1] and lower_bound[2] <= pixel_color[2] <= upper_bound[2]:
                    draw.line([center, (x, y)], fill=color_linea, width=1)
                    distancia = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                    resultado_operacion = -V + (distancia / radio_imagen) * V
                    resultados.append({
                        "angle": float(angle_deg),
                        "value": float(resultado_operacion)
                    })
                    break
    return image, resultados

def generar_plots_colores_interpolado_ampliado(imagen, colores_unicos, cantidad_a_mostrar=1, picos=None, ampliacion=15):
    """Genera una imagen interpolada basada en colores dominantes y dibuja círculos y líneas radiales."""
    if picos is None:
        raise ValueError("Se requiere la lista de picos del histograma.")
    color_hsv = colores_unicos[0]
    lower_bound = (int(color_hsv * 180) - ampliacion, 50, 50)
    upper_bound = (int(color_hsv * 180) + ampliacion, 255, 255)

    imagen_hsv = np.array(imagen.convert('HSV'))
    mask_color = np.all((imagen_hsv >= lower_bound) & (imagen_hsv <= upper_bound), axis=-1)
    imagen_color = Image.new('RGB', imagen.size)
    for x in range(imagen.size[0]):
        for y in range(imagen.size[1]):
            if mask_color[y, x]:
                imagen_color.putpixel((x, y), imagen.getpixel((x, y)))

    imagen_color = redimensionar_imagen_cuadrada(imagen_color)
    imagen_color = redimensionar_imagen(imagen_color, imagen_color.size[0])

    imagen_color, resultados_radiales = dibujar_lineas_radiales(imagen_color, lower_bound, upper_bound)

    centro = (imagen_color.size[0] // 2, imagen_color.size[1] // 2)
    valor_maximo_deseado = imagen_color.size[0] / 2
    logger.debug("Re-assigned M2: %d", M)
    factor_escala = valor_maximo_deseado / M
    radios = np.round(np.arange(0, M + N, N) * factor_escala).astype(int)
    imagen_con_circulos = dibujar_circulos_concentricos(imagen_color, centro, radios)

    return imagen_con_circulos, resultados_radiales

def process_image(image_data):
    """Main image processing function"""
    try:
        # Convert base64 to image
        image_data = image_data.split(';base64,')[-1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Validate image
        validate_image(image)

        # Convert to numpy array
        image_array = np.array(image)

        # Process histogram and peaks
        hist_hue = calcular_histograma_hue(image)
        picos = mostrar_histograma_con_picos(hist_hue)

        # Extract unique colors
        colores_unicos = mostrar_imagen_original_y_colores(image, picos)

        # Generate the interpolated image with concentric circles
        processed_image, vector_results = generar_plots_colores_interpolado_ampliado(
            image, 
            colores_unicos, 
            cantidad_a_mostrar=1, 
            picos=picos, 
            ampliacion=15
        )

        # Convert processed image to base64
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        processed_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Convert color peaks to HSV values for the response
        colores_dominantes = [{
            "h": float(pico),
            "s": 1.0,
            "v": 1.0,
            "rgb": mcolors.hsv_to_rgb([pico / 180.0, 1, 1]).tolist()
        } for pico in picos]

        response = {
            "processed_image": "data:image/png;base64," + processed_base64,
            "color_peaks": colores_dominantes,
            "vector_results": vector_results
        }

        return response

    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        raise

@app.route('/process-image', methods=['POST'])
def handle_image_processing():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        

        V = int(data['uv'])
        U = V - 3
        M = U - N
        
        # Log the new values of U and V
        logger.debug("Re-assigned U: %d", U)
        logger.debug("Re-assigned V: %d", V)
        logger.debug("Re-assigned M: %d", M)
        
        result = process_image(data['image'])
        return jsonify(result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
