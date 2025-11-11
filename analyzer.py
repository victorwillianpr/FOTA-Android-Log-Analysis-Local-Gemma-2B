import re
import json
from llama_cpp import Llama
from bs4 import BeautifulSoup

# --- Configuração do Modelo Local ---
MODEL_NAME = "gemma-2b-it.Q4_K_M.gguf"
MODEL_PATH = f"./models/{MODEL_NAME}"

try:
    print(f"Carregando o modelo local: {MODEL_PATH}...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=0, verbose=False)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"ERRO CRÍTICO AO CARREGAR O MODELO: {e}")
    print(f"Verifique se o arquivo '{MODEL_NAME}' está na pasta '/models'.")
    llm = None

def parse_log_stats(html_content: str) -> dict:
    # ... (função sem alterações)
    stats = {}
    try:
        match = re.search(r'window\.output\["stats"\]\s*=\s*(.*?);', html_content)
        if not match: return {}
        
        stats_data = json.loads(match.group(1))
        total_stats = stats_data[0][0] 
        
        passed = total_stats.get('pass', 0)
        failed = total_stats.get('fail', 0)
        skipped = total_stats.get('skip', 0)
        
        stats['passed'] = str(passed)
        stats['failed'] = str(failed)
        stats['skipped'] = str(skipped)
        stats['total'] = str(passed + failed + skipped)
    except Exception as e:
        print(f"Erro ao extrair estatísticas do log.html: {e}")
        return {}
    return stats

def parse_log_failed_details(html_content: str) -> list:
    # ... (função sem alterações)
    failed_tests = []
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        details_table = soup.find("table", id="test-details")
        if not details_table:
            return []

        fail_spans = details_table.find_all("span", class_="label fail", string="FAIL")
        
        for span in fail_spans:
            test_row = span.find_parent("tr")
            if test_row:
                test_name_tag = test_row.find("td", class_="details-col-name")
                test_msg_tag = test_row.find("td", class_="details-col-msg")

                if test_name_tag and test_msg_tag:
                    test_name = test_name_tag.get_text(strip=True)
                    error_message = test_msg_tag.get_text(strip=True)
                    
                    if not test_name.upper().startswith(('PASS', 'FAIL', 'SKIP')):
                          failed_tests.append({"name": test_name, "error": error_message})

    except Exception as e:
        print(f"Erro ao extrair detalhes de falha do log.html: {e}")
    
    unique_failed_tests = [dict(t) for t in {tuple(d.items()) for d in failed_tests}]
    return unique_failed_tests

def summarize_with_ai(stats: dict, failed_tests_details: list = None) -> str:
    if not llm:
        return "Erro: O modelo de linguagem local não foi carregado."
    if not stats:
        return "Nenhuma estatística de teste foi encontrada no log."

    try:
        passed = int(stats.get('passed', 0))
        failed = int(stats.get('failed', 0))
        total = int(stats.get('total', 0))
        accuracy = (passed / total) * 100 if total > 0 else 0
    except (ValueError, TypeError):
        return "Erro: Estatísticas de teste inválidas."

    # --- LÓGICA DE DECISÃO APRIMORADA ---
    # Agora, também verificamos os nomes dos testes que falharam para um veredito mais preciso.
    critical_tests_failed = False
    if failed_tests_details:
        critical_test_names = ["Conta Google", "Atualizar Sistema", "Validar Versão de Software", "Realizar Uma Ligação"]
        for test in failed_tests_details:
            if any(critical_name in test["name"] for critical_name in critical_test_names):
                critical_tests_failed = True
                break

    verdict_category = ""
    if failed == 0 and total > 0:
        verdict_category = "ESTAVEL"
    elif critical_tests_failed or accuracy < 60: # Se um teste crítico falhou OU a acurácia é muito baixa
        verdict_category = "CRITICO"
    else: # Falhas existem, mas não em testes críticos e a acurácia não é terrível
        verdict_category = "INSTAVEL"

    prompt_data = f"""
- Total de Casos de Teste: {total}
- Aprovados: {passed}
- Reprovados: {failed}
- Acurácia: {accuracy:.1f}%
"""
    if failed_tests_details:
        prompt_data += "\n### Detalhes das Falhas:\n"
        for test in failed_tests_details[:20]:
            prompt_data += f"- Teste: {test['name']}\n  - Erro: {test['error']}\n"

    user_prompt = ""
    
    if verdict_category == "ESTAVEL":
        user_prompt = f"""
        Você é um analista de testes FOTA. Os resultados dos testes foram EXCELENTES.
        
        **Dados da Execução:**
        {prompt_data}
        
        **Sua Tarefa:**
        Escreva uma análise técnica otimista.
        1.  **Resumo:** Apresente os números e a acurácia de 100%.
        2.  **Veredito da Build:** Declare a build como **"Build Estável"**.
        3.  **Análise de Impacto:** Afirme que, como todos os testes críticos de sistema, app e persistência de dados passaram, nenhum impacto negativo é esperado.
        4.  **Ação Recomendada:** Recomende **"Aprovar para a próxima fase de testes"**.
        """
    else: # Cenário INSTAVEL ou CRITICO
        if verdict_category == "INSTAVEL":
            verdict_text = '"Build Instável com Regressões"'
            action_text = '"Investigação prioritária das falhas é necessária. A build pode seguir para testes internos com ressalvas."'
        else: # CRITICO
            verdict_text = '"Build Inaceitável para Lançamento"'
            action_text = '"Bloquear o ciclo de release imediatamente. A correção das falhas críticas é obrigatória."'

        # --- PROMPT APRIMORADO COM CONHECIMENTO DOS TESTES ---
        user_prompt = f"""
        Você é um analista de testes FOTA. Os resultados dos testes apontaram falhas.

        **Instruções para a Análise de Impacto:**
        - Se a falha for em 'Conta Google', explique que isso impede o acesso à Play Store e serviços essenciais.
        - Se a falha for em 'Instalar' ou 'Atualizar' apps, mencione problemas na Play Store.
        - Se a falha for em 'Validar Histórico', 'Papel de Parede' ou 'Tela Inicial', aponte para a perda de dados e personalizações do usuário após a atualização.
        - Se a falha for em 'Validar Versão', questione se a atualização FOTA foi realmente bem-sucedida.
        - Se a falha for em 'Realizar Ligação', aponte para a falha da função mais básica do telefone.

        **Dados da Execução:**
        {prompt_data}

        **Sua Tarefa:**
        Escreva uma análise técnica de alerta.
        1.  **Resumo:** Apresente os números de aprovados, reprovados e a baixa acurácia.
        2.  **Veredito da Build:** Declare a build como **{verdict_text}**.
        3.  **Análise de Impacto:** **Foco principal aqui.** Usando as instruções acima, explique o que as falhas listadas nos "Detalhes das Falhas" significam para o usuário final. Seja específico e detalhado.
        4.  **Ação Recomendada:** Recomende **{action_text}**.
        """

    full_prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model"

    try:
        print(f"Gerando resumo (Veredito: {verdict_category})...")
        response = llm(
            full_prompt,
            max_tokens=1024,
            stop=["<end_of_turn>"],
            echo=False,
            temperature=0.9,
            repeat_penalty=1.1
        )
        print("Resumo gerado.")
        summary_header = f"**Resumo da Execução**\n- Testes Executados: {total}\n- Aprovados: {passed} ({accuracy:.1f}%)\n- Reprovados: {failed}\n\n---\n\n"
        ai_summary = response['choices'][0]['text'].strip()
        return summary_header + ai_summary
    except Exception as e:
        return f"Ocorreu um erro ao gerar o resumo com o modelo local: {e}"