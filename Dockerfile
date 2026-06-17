FROM python:3.10-slim

#diretório de trabalho
WORKDIR /app

#arquivos do projeto
COPY . .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta usada pelo Flask
EXPOSE 5001

# Comando para rodar a aplicação
CMD ["python", "src/rag/server.py"]
