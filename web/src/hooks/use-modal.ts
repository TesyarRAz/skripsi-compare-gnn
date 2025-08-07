import { create } from 'zustand'
// import { ModalContext } from '../providers/content/modal.context'

// const useModal = (name: string) => {
//     const context = useContext(ModalContext)

//     return useMemo(() => ({
//         openModal: context.openModal[name],
//         disabled: context.disabled[name],
//         isLoading: context.isLoading[name],
//         open: () => context.open(name),
//         close: () => context.close(name),
//         setDisable: (value = false) => context.setDisable(name, value),
//         setLoading: (value = false) => context.setLoading(name, value),
//         reset: () => context.reset()
//     }), [ context, name ])
// }

export interface ModalHandler {
    open: (name?: string) => void;
	close: (name?: string) => void;
	setDisable: (name?: string, value?: boolean) => void;
	setLoading: (name?: string, value?: boolean) => void;
	reset: () => void;
}

export interface ModalState {
	openModal: Record<string, boolean>;
	disabled: Record<string, boolean>;
	isLoading: Record<string, boolean>;
}


export const defaultState: ModalState = {
	openModal: {},
	disabled: {},
	isLoading: {},
};

const useModal = create<ModalState & ModalHandler>((set) => ({
	...defaultState,
	open(name) {
		if (name && !this.openModal[name]) {
			set({ openModal: { ...this.openModal, [name]: true } });
		}
	},
	close(name) {
		if (name && this.openModal[name]) {
			set({
				openModal: { ...this.openModal, [name]: false },
				disabled: { ...this.disabled, [name]: false },
				isLoading: { ...this.isLoading, [name]: false },
			});
		}
	},
	setDisable(name, value = false) {
		if (name && this.openModal[name] && this.disabled[name]) {
			set({ disabled: { ...this.disabled, [name]: value } });
		}
	},
	setLoading(name, value = false) {
		if (name && this.openModal[name]) {
			set({ isLoading:  { ...this.isLoading, [name]: value } });
		}
	},
	reset() {
		set(defaultState)
	},
}))

export default useModal;