# app.py

from flask import Flask, render_template, request
import analyzer 
import time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        log_file = request.files.get('log_file')

        if not log_file or log_file.filename == '':
            return render_template('index.html', error="Nenhum arquivo log.html foi selecionado.")

        if not log_file.filename.lower().endswith('log.html'):
            return render_template('index.html', error="Por favor, envie um arquivo log.html válido.")
        
        try:
            print("\n--- INICIANDO ANÁLISE DO LOG.HTML ---")
            
            log_content = log_file.stream.read().decode('utf-8')

            stats = analyzer.parse_log_stats(log_content)
            if not stats:
                return render_template('index.html', error="Não foi possível encontrar estatísticas no log.html.")
            print("Estatísticas gerais extraídas com sucesso.")

            failed_tests_details = analyzer.parse_log_failed_details(log_content)
            if failed_tests_details:
                print(f"Encontrados detalhes de {len(failed_tests_details)} testes que falharam.")

            start_ai = time.perf_counter()
            summary = analyzer.summarize_with_ai(stats, failed_tests_details)
            end_ai = time.perf_counter()
            ai_time = end_ai - start_ai
            print(f"Tempo de Geração da Análise (IA): {ai_time:.4f} segundos")

            print("--- ANÁLISE CONCLUÍDA ---")
            
            # --- MODIFICAÇÃO PRINCIPAL AQUI ---
            # Enviando o tempo de análise para o template
            return render_template(
                'index.html', 
                analysis_result=summary, 
                analysis_time=f"{ai_time:.2f}"
            )

        except Exception as e:
            return render_template('index.html', error=f"Ocorreu um erro inesperado: {e}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)