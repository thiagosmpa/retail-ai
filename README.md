## RETAIL AI - Análise Inteligente para o Varejo

**Aprimore a experiência de compra e otimize suas operações com o poder da Visão Computacional.**

![RETAIL AI](src/output.gif)

Este projeto utiliza tecnologias avançadas de IA para fornecer insights valiosos sobre o comportamento dos clientes em ambientes de varejo, como o fluxo de pessoas, tempo de permanência e até mesmo estimativas demográficas.

### Funcionalidades Principais:

* **Detecção e Rastreamento de Pessoas:** Identifica e acompanha o movimento de indivíduos em tempo real.
* **Estimativa de Gênero e Idade:** A partir da detecção facial, estima informações demográficas dos clientes.
* **Análise de Tempo de Permanência:** Calcula o tempo que cada pessoa passa em diferentes áreas da loja.
* **Heatmaps:** Gera mapas de calor para visualizar o fluxo de clientes e identificar zonas de maior interesse.
* **Logs Detalhados:** Registra informações cruciais em formatos JSON e CSV para análises posteriores.
* **Contagem de Entradas e Saídas:** Monitora o número de pessoas que entram e saem de áreas específicas.

### Execução de Modelos YOLO:

A função `run_yolo_models` simplifica a execução de modelos YOLO para detecção, rastreamento e estimativa de pose.

```python
def run_yolo_models(model_path, task, format, **kwargs):
    """
    Executa modelos YOLO automaticamente para detecção, rastreamento ou estimativa de pose.

    Args:
        model_path (str): Caminho para o arquivo do modelo.
        task (str): Tarefa a ser realizada pelo modelo ('track', 'detect' ou 'pose').
        format (str): Formato do arquivo do modelo ('openvino', 'onnx' ou 'pt').
        **kwargs: Se "classes" for passado, o modelo filtrará as detecções por classe.
    """
```

### Estrutura do Projeto:

* **`main.py`**: Responsável por executar a inferência completa do projeto em vídeos, aplicando todas as funcionalidades em conjunto para gerar análises abrangentes.
* **`visualize.ipynb`**: Notebook Jupyter que demonstra as técnicas individuais do projeto (detecção, rastreamento, estimativa de idade e gênero, etc.) de forma isolada em imagens estáticas, facilitando a compreensão de cada etapa do processo.

### Downloads e Repositórios:

* **Arquivos do Projeto (incluindo video original utilizado na inferência e os outputs relacionados):** [Retail files](https://drive.google.com/drive/folders/1XzXzfcilRSrZhu5I0jb4mRgxC1q4WJiP?usp=share_link)
* **Pesos e Checkpoints do Mivolo:** [Mivolo weights / mivolo checkpoint](https://drive.google.com/drive/folders/1FagDwoq8GfayuBLEye5IolINvF-9ixDO?usp=share_link)
* **Repositório Mivolo:** [mivolo model link](https://github.com/WildChlamydia/MiVOLO)

### Próximos Passos:

* **Instalação:** Detalhes sobre como configurar o ambiente e instalar as dependências necessárias.
* **Execução:** Instruções passo a passo para executar o projeto, incluindo exemplos de uso da função `run_yolo_models` e como utilizar `main.py` e `visualize.ipynb`.
* **Configuração:** Explicação de como personalizar o projeto para diferentes cenários e necessidades.
* **Contribuição:** Diretrizes para contribuir com o desenvolvimento do projeto.

**Com o RETAIL AI, transforme dados em decisões estratégicas para o seu negócio.**