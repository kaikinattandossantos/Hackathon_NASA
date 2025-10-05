import Logo from '../assets/logo.png'

const PopUpNotification = ({onClick, onClose}) => {
  return (
    <div className="flex fixed bottom-6 z-[9999] m-5">
      <div
      onClick={onClick}
      className="flex flex-row items-center justify-center gap-4 bg-white rounded-full shadow-2xl p-6 w-auto h-20 animate-fadeIn cursor-pointer">
        <img src={Logo} alt="Logo" className="w-auto h-[40px]" />
        <p className="text-gray-600 font-medium m-0">
          O assistente de IA tem uma sugestão pra você
        </p>
        <button
        onClick={(e) => {
          e.stopPropagation();
          onClose?.();
        }}
        className="bg-red-500 text-white rounded-full hover:bg-red-600 transition-all w-6 h-6 flex items-center justify-center">
          X
        </button>
      </div>
    </div>
  );
};

export default PopUpNotification;
