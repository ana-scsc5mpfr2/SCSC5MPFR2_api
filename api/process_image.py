from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
from PIL import Image
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
V = 15
N = 2
U = 12
M = U - N

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

def mostrar_histograma_con_picos(hist):
    """Encuentra los picos más altos y aplica suavizado al histograma."""
    picos = encontrar_tres_picos_mas_altos(hist)
    hist_reducido = cv2.GaussianBlur(hist.reshape(-1, 1), (5, 1), 0)
    return picos

def mostrar_imagen_original_y_colores(imagen_array, picos):
    """Muestra la imagen original y convierte los picos a valores de tono."""
    imagen_hsv = cv2.cvtColor(imagen_array, cv2.COLOR_RGB2HSV)
    tonos_picos = [pico / 180.0 for pico in picos]
    return tonos_picos

def redimensionar_imagen(imagen_array, tamano):
    """Redimensiona la imagen al tamaño especificado."""
    return cv2.resize(imagen_array, (tamano, tamano))

def redimensionar_imagen_cuadrada(imagen_array):
    """Redimensiona la imagen al tamaño de un cuadrado."""
    if imagen_array.shape[2] == 4:  # Si es RGBA
        imagen_array = cv2.cvtColor(imagen_array, cv2.COLOR_RGBA2RGB)

    altura, ancho, _ = imagen_array.shape
    tamano_maximo = max(altura, ancho)
    imagen_cuadrada = np.zeros((tamano_maximo, tamano_maximo, 3), dtype=imagen_array.dtype)
    imagen_cuadrada[:altura, :ancho] = imagen_array
    return imagen_cuadrada

def dibujar_circulos_concentricos(imagen_array, centro, radios):
    """Dibuja círculos concéntricos alrededor del centro dado."""
    imagen_con_circulos = imagen_array.copy()
    for radio in radios:
        cv2.circle(imagen_con_circulos, centro, radio, (0, 255, 0), 2)
    return imagen_con_circulos

def dibujar_lineas_radiales(image, num_lines=360, color_linea=(255, 0, 0)):
    """Dibuja líneas radiales desde el centro de la imagen."""
    h, w, _ = image.shape
    center = (w // 2, h // 2)
    radio_imagen = min(w, h) / 2

    resultados = []
    imagen_con_lineas = image.copy()

    for i in range(num_lines):
        angle = 2 * np.pi * i / num_lines + np.pi
        angle_deg = np.degrees(angle) % 360

        cos_angle, sin_angle = np.cos(angle), np.sin(angle)

        for r in range(int(radio_imagen)):
            x = int(center[0] + cos_angle * r)
            y = int(center[1] - sin_angle * r)

            if 0 <= x < w and 0 <= y < h and not np.all(image[y, x] == [0, 0, 0]):
                cv2.line(imagen_con_lineas, center, (x, y), color_linea, 1)
                distancia = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                resultado_operacion = -V + (distancia / radio_imagen) * V
                resultados.append({
                    "angle": float(angle_deg),
                    "value": float(resultado_operacion)
                })
                break

    return imagen_con_lineas, resultados

def generar_plots_colores_interpolado_ampliado(imagen_array, colores_unicos, cantidad_a_mostrar=1, picos=None, ampliacion=15):
    """Genera una imagen interpolada basada en colores dominantes y dibuja círculos y líneas radiales."""
    if picos is None:
        raise ValueError("Se requiere la lista de picos del histograma.")

    color_hsv = colores_unicos[0]
    color_rgb = np.array([mcolors.hsv_to_rgb([color_hsv, 1, 1])])

    lower_bound = np.array([int(color_hsv * 180) - ampliacion, 50, 50])
    upper_bound = np.array([int(color_hsv * 180) + ampliacion, 255, 255])

    imagen_hsv = cv2.cvtColor(imagen_array, cv2.COLOR_RGB2HSV)
    mask_color = cv2.inRange(imagen_hsv, lower_bound, upper_bound)
    imagen_color = cv2.bitwise_and(imagen_array, imagen_array, mask=mask_color)

    interpolated_image_rgb_cuadrada = redimensionar_imagen_cuadrada(imagen_color)
    interpolated_image_rgb_final = redimensionar_imagen(interpolated_image_rgb_cuadrada, interpolated_image_rgb_cuadrada.shape[0])

    # Dibujar líneas radiales
    interpolated_image_rgb_final, resultados_radiales = dibujar_lineas_radiales(interpolated_image_rgb_final)

    # Dibujar círculos concéntricos
    altura, ancho, _ = interpolated_image_rgb_final.shape
    centro = (ancho // 2, altura // 2)
    valor_maximo_deseado = ancho / 2

    factor_escala = valor_maximo_deseado / M
    radios = np.round(np.arange(0, M + N, N) * factor_escala).astype(int)

    imagen_con_circulos = dibujar_circulos_concentricos(interpolated_image_rgb_final, centro, radios)

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
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Procesar histograma y picos
        hist_hue = cv2.calcHist([cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)], [0], None, [180], [0, 180])
        picos = mostrar_histograma_con_picos(hist_hue)
        
        # Extraer colores únicos
        colores_unicos = mostrar_imagen_original_y_colores(image_array, picos)
        
        # Generar la imagen interpolada con círculos concéntricos
        processed_image, vector_results = generar_plots_colores_interpolado_ampliado(
            image_array, 
            colores_unicos, 
            cantidad_a_mostrar=1, 
            picos=picos, 
            ampliacion=15
        )
        
        # Convert processed image to base64
        success, buffer = cv2.imencode('.png', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Failed to encode processed image")
        
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Convert color peaks to HSV values for the response
        colores_dominantes = [{
            "h": float(pico),
            "s": 1.0,
            "v": 1.0,
            "rgb": mcolors.hsv_to_rgb([pico/180.0, 1, 1]).tolist()
        } for pico in picos]
        
        return {
            'processed_image': f'data:image/png;base64,{processed_base64}',
            'vector_results': vector_results,
            'dominant_colors': colores_dominantes
        }
    
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"Error processing image: {str(e)}")

@app.route('/api/process-image', methods=['POST'])
def handle_image_processing():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        result = process_image(data['image'])
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500

# Entry point for Vercel
def handler(event, context):
    from flask import Response
    return Response(app(event['body'], event['context']), mimetype='application/json')