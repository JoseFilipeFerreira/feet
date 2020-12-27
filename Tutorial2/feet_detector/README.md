Aplicação de métodos clássicos de processamento de imagem na detecção e localização dos pés

A partir de uma sequência de imagens/video capturadas com uma câmera RGBD (cor + profundidade - vídeo e imagens anexados, 'etc/gait_rgb.avi' e 'etc/gait_depth_frame.png'), é pretendido o desenvolvimento de um algoritmo para a detecção dos pés, particularmente as articulações do tornozelo e ponta do pé, utilizando para isso métodos e ferramentas de processamento de imagem clássico. A deteção em tempo real destes pontos de interesse é de particular relevância na avaliação da marcha de um determinado sujeito, visto permitir efetuar uma análise de marcha detalhada através de uma multitude de métricas espacio-temporais, tendo especial valor no seguimento da reabilitação de pacientes com limitações ao nível da marcha (p.e. casos clínicos de ataxia pós-enfarte).

O trabalho incidirá sobre a melhoria de uma solução funcional já implementada (código fornecido - exemplo de deteção no vídeo 'etc/feet_detection_example.avi'; algoritmo descrito no diagrama '/etc/foot_detection.pdf'), com acesso a um dataset construído a partir de ensaios já conduzidos num andarilho inteligente equipado com duas câmaras RGBD convencionais (tipo Kinect), em vários contextos (direcção/exposição luminosa), a várias velocidades de locomoção. Pretende-se que sejam identificadas e endereçadas as limitações da solução atual, e que seja feito desenvolvimento posterior de modo a tornar o método mais robusto e aproximar o erro de localização do valor referência (<= 3cm "state of the art"). Para isso irá também ser feita validação comparando com sistema de captura de movimento comercial (XSens). 

Descrição do código/solução atual: 

Desenvolvido em C++, a solução atual está centrada na classe asbgo::vision::FeetDetector (cf. FeetDetector.hpp/.cpp) que implementa todas as operações de processamento de imagem tanto separadamente (opcional, através de métodos estáticos) como em conjunto/sequencialmente (abordagem "black box"), fornecendo ainda métodos para visualização dos resultados da deteção; A parametrização do algoritmo é feita através de uma estrutura do tipo cv::FeetDetectorParameters, que permite alterar os vários parâmetros que influenciam o funcionamento/processo de deteção (e.g. dimensão de kerneis morfológicos, dimensões referência do sujeito, thresholds para profundidade e área de contorno, etc.). Um apontador para uma instância desta classe é passado para o constructor da classe cv::FeetDetector que por sua vez utiliza os parâmetros ao fazer a detecção.

Conceptualmente, esta abordagem é similar à adoptada por várias classes do género (cf. cv::SimpleBlobDetector, cv::ArucoDetector, etc.) incluídas nas bibliotecas OpenCV. A representação dos resultados é feita através da classe cv::Skeleton (cf. Skeleton.hpp/.cpp) que codifica as posições do esqueleto humano (modelo biomecânico do sistema XSens, imagem 'etc/xsens_joints.png'). No caso da detecção dos pés, apenas as articulações detectadas - "Foot" e "Toe" para cada pé - são preenchidas. Um exemplo de aplicação está fornecido no ficheiro 'example.cpp'.

Adicionalmente, é também fornecido código que permite o carregamento e sincronização em runtime do dataset construído sobre o qual incidirá o trabalho. Ao instanciar a classe asbgo::vision::TrialData (cf. TrialData.hpp), tanto os dados de cor (vídeos) como os dados de profundidade (imagens) são carregadas e os "time stamps" de cada amostra idenficados, sendo possível obter um conjunto de dados sincronizados através do método TrialData::next().

Nota: caso seja necessário, pode também ser fornecida a documentação da API (doxygen) para a solução desenvolvida.

Ficheiros anexados:

'include/' -> protótipos das classes referidas
'src/' -> implementação das classes referidas
'compile.sh' -> script para compilação
'etc/' -> ficheiros demonstrativos 
