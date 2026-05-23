from search import answer_question

def main():
    print("Chat iniciado. Digite sua pergunta ou 'sair' para encerrar.\n")

    while True:
        question = input("Pergunta: ").strip()
        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            return

        if not question:
            print("Digite uma pergunta válida.\n")
            continue

        try:
            answer = answer_question(question)
            print(f"\nResposta:\n{answer}\n")
        except Exception as exc:
            print(f"Erro ao responder a pergunta: {exc}\n")

if __name__ == "__main__":
    main()